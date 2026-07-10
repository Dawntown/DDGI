import os, itertools, functools, time
import torch
import tqdm
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
from typing import Iterable, Tuple
from .dataset import PairDataset
import torch.nn.functional as F

from ..myutils import is_verbose

geneinfo_alias = {
    'human': pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '../meta/geneinfo_alias_human.csv'
    )),
    'mouse': pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '../meta/geneinfo_alias_mouse.csv'
    )),
}

def load_gene_emb(emb_name: str,
                  gene_list: list) -> np.ndarray:
    """Load optional pretrained gene embeddings for the candidate genes."""
    embedding_files = {
        'genept_ada': (
            'GenePT_gene_embedding_ada_text.pickle',
            'GenePT embeddings from https://github.com/yiqunchen/GenePT'
        ),
        'scgpt': (
            'gene_embeddings_scgpt.pkl',
            'scGPT gene embeddings from https://github.com/bowang-lab/scGPT'
        ),
    }
    if emb_name not in embedding_files:
        raise ValueError(f"Unknown gene embedding: {emb_name}")

    file_name, source = embedding_files[emb_name]
    emb_path = os.path.join(os.path.dirname(__file__), '../meta', file_name)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Optional embedding file not found: {emb_path}. "
            f"Download {source} and place it at driver_genes/meta/{file_name}, "
            f"or set identifier.pe_type='none'."
        )

    if is_verbose():
        print(f"Loading {emb_name} from {emb_path}")
    emb_dict = pd.read_pickle(emb_path)

    missing = sorted(set(gene_list) - set(emb_dict))
    if missing:
        preview = ', '.join(missing[:10])
        suffix = '...' if len(missing) > 10 else ''
        raise KeyError(f"{emb_name} is missing {len(missing)} candidate genes: {preview}{suffix}")

    if is_verbose():
        print(f"Loaded {len(gene_list)} gene embeddings from {emb_name}.")

    return torch.Tensor([emb_dict[gene] for gene in gene_list])
      


def prepare_datasets(adata_path: str | list[str], 
                     split_path: str | list[str], 
                     layer_name: str,
                     var_subset: str = 'highly_variable',
                     sample_size: int = 32,
                     sample_neg: bool = False,
                     cotrain_mode: bool = False,
                     batch_key: str = None,
                     condition_key: str = 'perturbation',
                     source_cell: str = 'control',
                     resampling_mode: bool = True,
                     ) -> Tuple[dict[str, dict[str, PairDataset]], np.ndarray, pd.DataFrame, np.ndarray, str, list]:
    # assert type(adata_path) == type(split_path), \
    #     "adata_path and split_path should be strs or lists at the same time"
    
    if isinstance(adata_path, str) and isinstance(split_path, str):
        adata_path = [adata_path]
        split_path = [split_path]
    elif isinstance(adata_path, str) and isinstance(split_path, list):
        adata_path = [adata_path] * len(split_path)
    elif isinstance(adata_path, list) and isinstance(split_path, str):
        split_path = [split_path] * len(adata_path)
    elif isinstance(adata_path, list) and isinstance(split_path, list):
        pass
    else:
        raise ValueError("Invalid adata_path and split_path")
    
    ad_list, sp_list, pgs_list, organism_list, batchs_list = [], [], [], [], []
    for ad_path, sp_path in zip(adata_path, split_path):
        ad, sp_df, pgs, organism, batchs = prepare_adata(
            adata_path=ad_path, 
            split_path=sp_path, 
            layer_name=layer_name, 
            var_subset=var_subset, 
            batch_key=batch_key,
            condition_key=condition_key,
            source_cell=source_cell
        )
        ad_list.append(ad)
        sp_list.append(sp_df)
        pgs_list.append(pgs)
        organism_list.append(organism)
        if batchs is not None:
            batchs_list.append(batchs)
    assert len(set(organism_list)) == 1, \
        f"All adata should have the same organism: {organism_list}"
    pertgenes = functools.reduce(
        np.union1d, 
        pgs_list, 
        pgs_list[0]
    )
    if batchs_list:
        batchs = functools.reduce(
            np.union1d, 
            batchs_list, 
            batchs_list[0]
        )
        if is_verbose():
            print(f"Using {len(batchs)} batches: {batchs}")
    else:
        batchs = None
    
    var_order = functools.reduce(
        np.intersect1d, 
        [ad.var_names for ad in ad_list], 
        ad_list[0].var_names
    )
    if len(var_order) != len(ad_list[0].var_names):
        n_var_list = [len(ad.var_names) for ad in ad_list]
        Warning(f"The kept genes are less than the original genes: {n_var_list} -> {len(var_order)}")
    
    # reorder adata's genes to let them have the same feature order
    ad_list = [ad[:, var_order] for ad in ad_list]
    
    ds_dict = {} # split keyed dict of named adata
    
    for ad, sp, ad_path in zip(ad_list, sp_list, adata_path):
        ad_name = os.path.splitext(os.path.basename(ad_path))[0]
        ad_dict = split_adata(ad, sp, cotrain_mode=cotrain_mode)
        for split, ad_dict_split in ad_dict.items():
            if split == 'cotrain' and cotrain_mode:
                if is_verbose():
                    print(f"Adding cotrain dataset for {ad_name} to train set")
                ds_dict['train'] = ds_dict.get('train', {})
                ds_dict['train'].update({
                    f"{ad_name}_{subsplit}_cotrain": PairDataset(
                        adata=ad, 
                        pertgene_list=pertgenes, 
                        condition_key=condition_key,
                        source_cell=source_cell,
                        sample_size=sample_size, 
                        resampling_mode=resampling_mode,
                        batch_list=batchs, 
                        batch_key=batch_key, 
                        organism=organism_list[0],
                        sample_neg=sample_neg
                    )
                    for subsplit, ad in ad_dict_split.items()
                })
                continue
            ds_dict[split] = ds_dict.get(split, {})
            if split == 'pred':
                # For prediction, use non-resampling mode to get all target cells
                ds_dict[split].update({
                f"{ad_name}_{subsplit}": PairDataset(
                    adata=ad, 
                    pertgene_list=pertgenes, 
                    condition_key=condition_key,
                    source_cell=source_cell,
                    sample_size=sample_size, 
                    resampling_mode=False,  # Non-resampling for prediction
                    batch_list=batchs, 
                    batch_key=batch_key, 
                    organism=organism_list[0],
                    sample_neg=False
                )
                for subsplit, ad in ad_dict_split.items()
            })
            else:
                # For train/val/test, use resampling_mode from parameter
                ds_dict[split].update({
                    f"{ad_name}_{subsplit}": PairDataset(
                        adata=ad, 
                        pertgene_list=pertgenes, 
                        condition_key=condition_key,
                        source_cell=source_cell,
                        sample_size=sample_size, 
                        resampling_mode=resampling_mode,
                        batch_list=batchs, 
                        batch_key=batch_key, 
                        organism=organism_list[0],
                        sample_neg=sample_neg
                    )
                    for subsplit, ad in ad_dict_split.items()
                })
            
    return ds_dict, pertgenes, pd.concat(sp_list, axis=0), var_order, organism_list[0], batchs
        

def prepare_adata(adata_path: str, 
                  split_path: str, 
                  layer_name: str = 'X',
                  var_subset: str = 'highly_variable',
                  batch_key: str = None,
                  condition_key: str = 'perturbation',
                  source_cell: str = 'control',
                  ) -> Tuple[sc.AnnData, pd.DataFrame, np.ndarray, str, list]:
    """Prepare AnnData object for dataset creation.
    
    Args:
        adata_path: Path to h5ad file
        split_path: Path to split CSV file
        layer_name: Layer name to use (e.g., 'X' or 'raw')
        var_subset: Variable subset to use (e.g., 'highly_variable' or 'all')
        batch_key: Key in adata.obs for batch labels
        condition_key: Key in adata.obs for condition labels
        source_cell: Value in condition_key column for source cells
        
    Returns:
        Tuple of (adata, split_df, pertgenes, organism, batchs)
    """
    if is_verbose():
        print(f"Loading adata from {os.path.basename(adata_path)}")
    adata = sc.read_h5ad(adata_path, backed='r')
    adata.obs_names_make_unique()
    
    if split_path.endswith(('.csv', '.csv.gz')):
        split_df = pd.read_csv(split_path)
    elif split_path.endswith('.parquet'):
        split_df = pd.read_parquet(split_path).reset_index(names='cell')
        if condition_key in split_df.columns:
            split_df = split_df.drop(columns=condition_key)
        
    if 'cell' in split_df.columns:
        adata = adata[adata.obs_names.isin(
            split_df['cell'].values
        )].to_memory()
        split_df = split_df.loc[
            split_df['cell'].isin(adata.obs_names)
        ]
    elif condition_key in split_df.columns:
        # Filter by condition_key values in split_df
        source_cell_list = [source_cell] if source_cell not in split_df[condition_key].values else []
        condition_values = np.concatenate([
            split_df[condition_key].values, 
            source_cell_list
        ])
        adata = adata[adata.obs[condition_key].isin(condition_values)].to_memory()
        split_df = split_df.loc[
            split_df[condition_key].isin(adata.obs[condition_key].unique())
        ]
    else:
        raise ValueError("Unknown split dataframe.")
    split_df['adata_path'] = adata_path

    if var_subset == 'all':
        if is_verbose():
            print('Using all genes')
    else:
        adata = adata[:, adata.var[var_subset]]
        if is_verbose():
            print(f'Using {var_subset} genes: {len(adata.var_names)}')
    
    if layer_name in adata.layers.keys():
        adata.X = adata.layers[layer_name]
        if is_verbose():
            print(f'Using adata.layers["{layer_name}"] as the input matrix')
    else:
        if is_verbose():
            print('Using adata.X as the input matrix')
    
    # Extract pertgenes from condition labels
    # Only for perturbation-type conditions (gene1_gene2 format)
    perts = adata.obs[condition_key].unique()
    pertgenes = np.array(sorted(list(set([
        g for gs in perts 
        for g in gs.split('_') 
        if g not in ['control', 'neg', 'unknown']
    ]))))
    
    organism = adata.uns['organism'] if 'organism' in adata.uns.keys() else 'human'
    if batch_key in adata.obs.columns:
        batchs = adata.obs[batch_key].unique()
    elif batch_key is None or batch_key == 'None':
        batchs = None
    else:
        if is_verbose():
            print(f"Batch key {batch_key} is not in adata.obs.columns, using default batch 0")
        batchs = [0]
    
    if is_verbose():
        print(f"Splitting adata according to {os.path.basename(split_path)}")
    return adata, split_df, pertgenes, organism, batchs


def split_adata(adata: sc.AnnData, 
                split_df: pd.DataFrame, 
                cotrain_mode: bool = False,
                ) -> dict[str, dict[str, sc.AnnData]]:
    """
    Return a dictionary of AnnData objects, keyed by split in split_df.
    If cotrain_mode is True, the cotrain dataset is the test set.
    """
    if cotrain_mode:
        split_df = pd.concat([
            split_df,
            split_df.query("split == 'test'").assign(split='cotrain'),
            split_df.query("split == 'pred'").assign(split='test')
        ])
    
    if 'perturbation' in split_df.columns:
        if 'subsplit' in split_df.columns:
            Warning("subsplit is not supported for perturbation split")
        ps_ss = split_df.groupby('split', observed=True)['perturbation'].apply(list)
        ad_dict = {}
        for split, perts in ps_ss.to_dict().items():
            ad_dict[split] = {
                'all': adata[adata.obs['perturbation'].isin(perts + ['control'])]
            }
    else:
        if 'subsplit' in split_df.columns:
            split_key = ['split', 'subsplit']
        else:
            split_key = ['split']
        bcs_ss = split_df.groupby(split_key, observed=True)['cell'].apply(list)
        ad_dict = {}
        for split, bcs in bcs_ss.to_dict().items():
            if isinstance(split, tuple):
                split, subsplit = split
            else:
                subsplit = 'all'
            ad_dict[split] = ad_dict.get(split, {})
            ad_dict[split][subsplit] = adata[adata.obs_names.isin(bcs)]
    return ad_dict


def convert_alias_to_symbols(aliases, organism, verbose=True):
    """Convert aliases to official gene symbols,
    Translated from the R code of NicheNetR::supporting_functions.R.
    """
    if not isinstance(aliases, Iterable):
        raise ValueError("aliases should be a list of gene symbols")

    if organism not in geneinfo_alias:
        raise ValueError("Organism must be either 'human' or 'mouse'")

    geneinfo_df = geneinfo_alias[organism]

    orphan_aliases = list(set(aliases) - set(geneinfo_df['alias']))
    if orphan_aliases:
        if verbose:
            print(f"There are provided symbols for {organism} that are not in the alias annotation table:")
            print(orphan_aliases)
        orphan_alias_tbl = pd.DataFrame({'symbol': orphan_aliases, 'entrez': [None] * len(orphan_aliases), 'alias': orphan_aliases})
        # Update the DataFrame in the dictionary (or create a copy if modification is not desired globally)
        geneinfo_alias[organism] = pd.concat([geneinfo_df, orphan_alias_tbl], ignore_index=True)
        geneinfo_df = geneinfo_alias[organism] # Use the updated DataFrame
        if verbose:
            print("They are added to the alias annotation table, so they don't get lost")

    alias2symbol = dict(zip(geneinfo_df['alias'], geneinfo_df['symbol']))
    converted_symbols = [alias2symbol.get(alias, alias) for alias in aliases]
    changed_aliases = list(set(aliases) - set(converted_symbols))

    if verbose:
        if not changed_aliases:
            print(f"All input symbols for {organism} were official symbols")
        else:
            print(f"Following are the official gene symbols of input aliases for {organism}:")
            print(geneinfo_df[geneinfo_df['alias'].isin(changed_aliases)][['symbol', 'alias']])

    return converted_symbols


def convert_newnames(feature_names, organism, verbose=False):
    """Convert aliases to official gene symbols, but handling duplicates.
    This function is a wrapper around convert_alias_to_symbols.
    Translated from the R code of NicheNetR::supporting_functions.R.
    """
    newnames = convert_alias_to_symbols(feature_names, organism, verbose=verbose)

    # Handle duplicates
    duplicates = [name for name, count in pd.Series(newnames).value_counts().items() if count > 1]
    genes_remove = [feature_names[i] for i, name in enumerate(newnames) if name in duplicates and feature_names[i] != name]
    for gene in genes_remove:
        newnames[feature_names.index(gene)] = gene  # Set the duplicates back to their old names
    return newnames


def load_prior_data(output_dir: str, pe_type: str = 'none', organism: str = 'human'):
    candidate_genes = np.loadtxt(
        os.path.join(output_dir, 'pertgenes.txt'),
        dtype=str,
        ndmin=1
    )
    if pe_type == 'none':
        return candidate_genes, None
    if pe_type not in {'genept_ada', 'scgpt'}:
        raise ValueError(
            "Unsupported pe_type. DDGI supports only 'none', 'genept_ada', and 'scgpt'. "
            "Graph positional encodings such as MagLap are not included in this release."
        )
    prior_data = load_gene_emb(
        emb_name=pe_type,
        gene_list=candidate_genes,
    )
    return candidate_genes, prior_data

def load_batchs(output_dir: str):
    batchs = np.loadtxt(
        os.path.join(output_dir, 'batchs.txt'),
        dtype=str,
        ndmin=1
    )
    return batchs


def agg_preds(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Aggregate duplicated source and target cells in prediction outputs.

    During repeated prediction, the same cell can appear multiple times because
    control/target pairs are resampled. This function averages duplicated
    entries by barcode before reassembling predictions into AnnData.
    """
    aggs = {}

    x_bc_dedup = (
        pd.DataFrame({'bc': preds['x_bc']})
        .reset_index(names='idx')
        .groupby('bc')
        .agg(func=list)['idx']
    )
    for field in ['x_hat', 'zx', 'x']:
        aggs[field] = np.concatenate([
            preds[field][idx].mean(axis=0, keepdims=True)
            for idx in x_bc_dedup.values
        ], axis=0)
    aggs['x_bc'] = x_bc_dedup.index

    y_bc_dedup = (
        pd.DataFrame({'bc': preds['y_bc']})
        .reset_index(names='idx')
        .groupby('bc')
        .agg(func=list)['idx']
    )
    for field in ['y_hat', 'zy', 'sy', 'y', 'cls', 'psi', 'dy']:
        aggs[field] = np.concatenate([
            preds[field][idx].mean(axis=0, keepdims=True)
            for idx in y_bc_dedup.values
        ], axis=0)
    aggs['y_bc'] = y_bc_dedup.index

    return aggs


def reassemble_anndata(ds_dict: dict[str, PairDataset], pred_dict: dict[str, dict[str, np.ndarray]], agg: bool = True):
    """
    Input:
        ds_dict: dict(ad_name: adata)
        pred_dict: dict(ad_name: pred_dict)
    Output:
        adata: anndata
    """
    common_keys = set(ds_dict.keys()) & set(pred_dict.keys())
    ad_dict = {}
    for ad_name in common_keys:
        preds = pred_dict[ad_name]
        if agg:
            preds = agg_preds(preds)
        pertgenes = ds_dict[ad_name].pertgenes
        bc = np.concatenate([preds['x_bc'], preds['y_bc']], axis=0)
        ad = ds_dict[ad_name].ad[bc]
        ad.layers['X_hat'] = np.concatenate([preds['x_hat'], preds['y_hat']], axis=0)
        ad.layers['dy'] = np.concatenate([np.zeros_like(preds['x_hat']), preds['dy']], axis=0)
        ad.obsm['latent'] = np.concatenate([preds['zx'], preds['zy']], axis=0)
        ad.obsm['shift'] = np.concatenate([np.zeros_like(preds['zx']), preds['sy']], axis=0)
        ad.obsm['proba'] = pd.DataFrame(
            np.concatenate([np.zeros((len(preds['x_bc']), len(pertgenes))), preds['cls']], axis=0),
            index=bc,
            columns=pertgenes
        )
        ad.obs['bc'] = bc
        ad.obs_names_make_unique()
        ad.uns['pertgenes'] = pertgenes
        ad_dict[ad_name] = ad
    return ad_dict


def concat_results(results_list: list[dict]):
    """
    Input:
        results_list: list[dict(ad_name: dict(metric_name: metric_value))]
    Output:
        ad_results: dict(ad_name: dict(metric_name: tensor(list[metric_value])))
    """
    ad_results = {}
    for results_dict in results_list:
        for ad_name, results_sub in results_dict.items():
            ad_results[ad_name] = ad_results.get(ad_name, {})
            for k, v in results_sub.items():
                ad_results[ad_name][k] = ad_results[ad_name].get(k, []) + [v]
            
    for ad_name, results_sub in ad_results.items():
        for k, v in results_sub.items():
            if isinstance(v[0], torch.Tensor):
                ad_results[ad_name][k] = torch.concat(v).detach().cpu().numpy()
            else:
                ad_results[ad_name][k] = np.concatenate(v)

    return ad_results


def compute_jacobian(model, x, return_radio=False):
    """
    Compute the Jacobian matrix of the model's output with respect to its input.

    Args:
        model: A function or module that takes input of shape (batch, input_dim) and outputs (batch, output_dim).
        x: Input tensor of shape (batch, input_dim). Must have requires_grad=True.

    Returns:
        jacobian: Jacobian matrix of shape (batch, input_dim, output_dim)
    """
    batch_size, input_dim = x.shape
    x = x.clone().detach().requires_grad_(True)
    y = model(x)  # (batch, output_dim)
    output_dim = y.shape[1]
    
    jacobian = torch.zeros(batch_size, input_dim, output_dim, device=x.device)
    
    for i in range(output_dim):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, i] = 1.0
        
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True if i < output_dim - 1 else False
        )[0]
        jacobian[:, :, i] = grad
    
    if return_radio:
        return jacobian / y.unsqueeze(1)
    else:
        return jacobian


def fetch_latent2gene_grad(model, inputs, device='cuda', return_radio=False):
    model.eval()
    model.to(device)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    results = model(inputs)
    def sub_module(zx):
        if model.extractor.decoder.__class__.__name__ == 'ZINBDecoder':
            return model.extractor.decoder(zx)[0]
        else:
            return model.extractor.decoder(zx)
    jacobian = compute_jacobian(sub_module, results['zx'], return_radio=return_radio)
    return jacobian



def compute_pred(adata_pred: sc.AnnData, neg_thr: float = None):
    assert 'proba' in adata_pred.obsm.keys(), "Proba matrix is not in adata_pred.obsm"
    adata_pred.obsm['rank'] = adata_pred.obsm['proba'].rank(axis=1, ascending=False)
    adata_pred.obsm['RelPos'] = (adata_pred.obsm['proba'].shape[1] - adata_pred.obsm['rank'] + 1) / adata_pred.obsm['proba'].shape[1]
    adata_pred.obs['pred_perturbation'] = adata_pred.obsm['proba'].columns[adata_pred.obsm['proba'].values.argmax(axis=1)]
    adata_pred.obsm['logit'] = adata_pred.obsm['proba'].transform(lambda x: np.log((x + 1e-8) / (1 - x + 1e-8)))
    if neg_thr is not None:
        neg_mask = adata_pred.obsm['proba'].max(axis=1).values < neg_thr
        adata_pred.obs.loc[neg_mask, 'pred_perturbation'] = 'neg'
    return adata_pred
    

def compute_sim(
    expr1: np.ndarray, expr2: np.ndarray, 
    label1: np.ndarray, label2: np.ndarray, 
    index1: np.ndarray = None, index2: np.ndarray = None,
    method: str = 'cosine',
    device: str = 'cuda'
):
    """
    Compute average cosine similarity between all pairs of groups in label1 x label2.
    Returns a DataFrame where rows correspond to unique labels in label1,
    columns to unique labels in label2, and values are mean pairwise cosine similarity.
    """
    # Calculate full similarity matrix
    X = torch.from_numpy(np.array(expr1)).to(device)
    Y = torch.from_numpy(np.array(expr2)).to(device)
    if method == 'cosine':
        sim_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2).cpu().numpy()
    elif method == 'pearson':
        sim_matrix = _pearson_corr(X, Y).cpu().numpy()
    elif method == 'spearman':
        sim_matrix = _spearman_corr(X, Y).cpu().numpy()
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    if not(index1 is None) and not(index2 is None):
        # Set similarity of the same obs (with the same index) to NaN
        common = np.intersect1d(index1, index2)
        for idx in common:
            i1 = np.where(index1 == idx)[0]
            i2 = np.where(index2 == idx)[0]
            for k in i1:
                for l in i2:
                    sim_matrix[k, l] = np.nan
    
    # Prepare output DataFrame
    unique_label1 = np.unique(label1)
    unique_label2 = np.unique(label2)
    output = np.zeros((len(unique_label1), len(unique_label2)))
    
    for i, l1 in enumerate(unique_label1):
        idx1 = np.where(label1 == l1)[0]
        for j, l2 in enumerate(unique_label2):
            idx2 = np.where(label2 == l2)[0]
            # Mean of the submatrix corresponding to (l1, l2)
            if idx1.size > 0 and idx2.size > 0:
                output[i, j] = np.nanmean(sim_matrix[np.ix_(idx1, idx2)])
            else:
                output[i, j] = np.nan
    sim_df = pd.DataFrame(output, index=unique_label1, columns=unique_label2)
    return sim_df


def _pearson_corr(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # Center X and Y along features (dim=1)
        X_centered = X - X.mean(dim=1, keepdim=True)
        Y_centered = Y - Y.mean(dim=1, keepdim=True)
        # Calculate norms
        X_norm = torch.norm(X_centered, dim=1, keepdim=True) + 1e-8
        Y_norm = torch.norm(Y_centered, dim=1, keepdim=True) + 1e-8
        # Normalize X and Y
        X_normed = X_centered / X_norm
        Y_normed = Y_centered / Y_norm
        # Pearson similarity is just the cosine similarity of mean-centered vecs
        sim_matrix = torch.matmul(X_normed, Y_normed.t())
        return sim_matrix
    

def _spearman_corr(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Spearman correlation matrix between rows of X and rows of Y.
    Uses average ranks for ties (true Spearman behavior), implemented in Torch.
    """

    def _rankdata_average_ties(A: torch.Tensor) -> torch.Tensor:
        # Rank each row along dim=1 with average rank for ties.
        # Returns 1-based ranks as float tensor with same shape as A.
        A = A.float()
        n, d = A.shape
        ranks = torch.empty_like(A)

        for i in range(n):
            vals = A[i]
            # stable sort for deterministic behavior
            sorted_vals, sorted_idx = torch.sort(vals, stable=True)

            # Identify tie-group starts
            change = torch.ones(d, dtype=torch.bool, device=A.device)
            if d > 1:
                change[1:] = sorted_vals[1:] != sorted_vals[:-1]

            starts = torch.nonzero(change, as_tuple=False).squeeze(1)
            ends = torch.cat([starts[1:], torch.tensor([d], device=A.device)])

            ranks_sorted = torch.empty(d, dtype=A.dtype, device=A.device)
            for s, e in zip(starts.tolist(), ends.tolist()):
                # average of 1-based positions [s+1, ..., e]
                avg_rank = 0.5 * ((s + 1) + e)
                ranks_sorted[s:e] = avg_rank

            # unsort back to original order
            ranks[i, sorted_idx] = ranks_sorted

        return ranks

    X_rank = _rankdata_average_ties(X)
    Y_rank = _rankdata_average_ties(Y)

    # Spearman = Pearson on rank-transformed data
    return _pearson_corr(X_rank, Y_rank)
