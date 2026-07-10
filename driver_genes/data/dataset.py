import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import Dataset
from typing import Iterable, Optional

class PairDataset(Dataset):
    """Dataset for pairing control and perturbed cells.
    
    Supports resampling mode for training/evaluation and non-resampling mode for prediction.
    Can be configured with different condition keys (e.g., 'perturbation' or 'cell_type') 
    and source cell types (e.g., 'control' or 'Monocyte').
    
    Args:
        adata: AnnData object containing the data
        pertgene_list: List of perturbation gene names
        condition_key: Key in adata.obs for condition labels (e.g., 'perturbation' or 'cell_type')
        source_cell: Value in condition_key column for source cells (e.g., 'control' or 'Monocyte')
        sample_size: Number of samples per condition for resampling mode (None for non-resampling)
        resampling_mode: If True, resample pairs for training/evaluation. If False, use all pairs for prediction.
        batch_list: List of batch names (None if no batch effect)
        batch_key: Key in adata.obs for batch labels
        sample_neg: Whether to sample negative perturbation pairs
        **kwargs: Additional arguments (e.g., organism, pert_type)
    """
    def __init__(self, 
                 adata: sc.AnnData, 
                 pertgene_list: Iterable = None, 
                 condition_key: str = 'perturbation',
                 source_cell: str = 'control',
                 resampling_mode: bool = True,
                 sample_size: int = None,
                 batch_list: Iterable = None,
                 batch_key: str = None,
                 sample_neg: bool = False,
                 **kwargs,
                ):
        self.ad = adata
        self.pertgenes = pertgene_list
        self.condition_key = condition_key
        self.source_cell = source_cell
        self.resampling_mode = resampling_mode
        self.sample_size = sample_size
        # allow specifying a reference batch (for padding_idx=0 scheme)
        self.batch_ref = kwargs.get('batch_ref')
        if batch_list is not None and len(batch_list) > 1:
            if self.batch_ref is not None and self.batch_ref in batch_list:
                # move reference batch to the front so its index = 0
                batch_list = [self.batch_ref] + [b for b in batch_list if b != self.batch_ref]
            self.batchs = batch_list
        else:
            self.batchs = None
        self.batch_key = batch_key
        self.sample_neg = sample_neg

        
        # Validate condition_key and source_cell
        assert condition_key in adata.obs.columns, f"condition_key '{condition_key}' not found in adata.obs"
        assert source_cell in adata.obs[condition_key].unique(), \
            f"source_cell '{source_cell}' not found in {condition_key} column"
        
        # Validate pertgene_list
        if pertgene_list is not None:
            assert source_cell not in pertgene_list, \
                f"source_cell '{source_cell}' should not be in pertgene_list"
        
        if sample_neg:
            self.ad = self._add_negative_pert(self.ad, condition_key, source_cell)
        
        self.X = self.ad.X if isinstance(self.ad.X, np.ndarray) else self.ad.X.toarray()
        self.X = torch.Tensor(self.X)
        
        self.obs_df = self.ad.obs.copy()
        
        # Get all conditions except source_cell
        self.perts = np.array([
            p for p in self.obs_df[condition_key].unique() 
            if p != source_cell
        ])
        self.pert2vec = {
            pgs: torch.LongTensor(np.isin(self.pertgenes, pgs.split('_')).astype(int)) 
            for pgs in self.perts
        }
        
        if self.batchs is not None:
            self.batch2idx = {b: i for i, b in enumerate(self.batchs)}
            self.obs_df['b'] = self.obs_df[self.batch_key]
        
        self.obs_df['bc'] = self.obs_df.index
        self.obs_df['idx'] = np.arange(self.obs_df.shape[0])
        self.obs_df.index = self.obs_df['idx'].values
        self.all_idx_pert_dict = self.obs_df.groupby(
            condition_key, observed=True
        )['idx'].apply(list).to_dict()
                
        self.rng = torch.Generator()
        self.initialize()
        
        # Other auxiliary info
        self.organism = kwargs.get('organism', 'unknown')
        self.pert_type = kwargs.get('pert_type', 'unknown')
        
    def sample_idx(self, perturb: str = None):
        """Sample indices for a given condition.
        
        Args:
            condition: Condition name. If None, uses source_cell.
            
        Returns:
            Tensor of sampled indices
        """
        if perturb is None:
            perturb = self.source_cell
        
        indices = torch.tensor(self.all_idx_pert_dict[perturb])
        
        if perturb == self.source_cell:
            if self.resampling_mode:
                num_samples = self.sample_size * len(self.perts)
            else:
                num_samples = sum(len(self.all_idx_pert_dict[pert]) for pert in self.perts if pert != self.source_cell)
        else:
            if self.resampling_mode:
                num_samples = self.sample_size
            else:
                num_samples = len(indices)
        
        if len(indices) < num_samples:
            # With replacement if not enough samples
            return indices[torch.randint(
                0, len(indices), 
                (num_samples,), 
                generator=self.rng
            )]
        else:
            # Without replacement if enough samples
            return indices[torch.randperm(
                len(indices), 
                generator=self.rng
            )[:num_samples]]
    
    def initialize(self, seed=None):
        """Initialize the dataset with random pairs.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            self.rng.manual_seed(seed)
        
        self.source_indices = self.sample_idx(self.source_cell)
        self.target_indices = torch.cat([
            self.sample_idx(pert) for pert in self.perts
        ])
        self.target_indices = self.target_indices[torch.randperm(len(self.target_indices), generator=self.rng)]
        
    @staticmethod
    def _add_negative_pert(adata: sc.AnnData, condition_key: str = 'perturbation', source_cell: str = 'control') -> sc.AnnData:
        """Add negative perturbation samples by sampling from control cells.
        
        Args:
            adata: AnnData object
            condition_key: Key in adata.obs for condition labels
            
        Returns:
            Modified AnnData object with negative samples
        """
        if not isinstance(adata.obs[condition_key].dtype, pd.CategoricalDtype):
            adata.obs[condition_key] = adata.obs[condition_key].astype('category')
        
        all_categories = list(adata.obs[condition_key].cat.categories)
        
        if 'neg' not in all_categories:
            # Add neg category
            all_categories.append('neg')
            adata.obs[condition_key] = adata.obs[condition_key].cat.set_categories(all_categories)
            # Sample neg from source cells
            source_bcs = adata.obs[adata.obs[condition_key] == source_cell].index.values
            neg_bcs = source_bcs[torch.randperm(len(source_bcs))[:len(source_bcs) // 5]]
            adata.obs.loc[neg_bcs, condition_key] = 'neg'
        return adata

    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.target_indices)
        
    def __getitem__(self, i):
        """Get a pair of cells at index i.
        
        Returns:
            Dictionary containing:
                - x: Source cell expression
                - y: Target cell expression
                - psi: Condition vector (perturbation genes)
                - x_bc: Source cell barcode
                - y_bc: Target cell barcode
                - x_batch: Source cell batch (if applicable)
                - y_batch: Target cell batch (if applicable)
        """
        idx_x = self.source_indices[i]
        idx_y = self.target_indices[i]
        condition = self.obs_df.loc[idx_y.item(), self.condition_key]
        psi = self.pert2vec[condition]
        x = self.X[idx_x]
        y = self.X[idx_y]
        
        inputs = {
            'x': x, 
            'y': y, 
            'psi': psi,
            'x_bc': self.obs_df.loc[idx_x.item(), 'bc'],
            'y_bc': self.obs_df.loc[idx_y.item(), 'bc'],
        }
        
        if self.batchs is not None:
            x_batch = self.batch2idx[self.obs_df.loc[idx_x.item(), 'b']]
            y_batch = self.batch2idx[self.obs_df.loc[idx_y.item(), 'b']]
            inputs['x_batch'] = x_batch
            inputs['y_batch'] = y_batch
        
        return inputs

