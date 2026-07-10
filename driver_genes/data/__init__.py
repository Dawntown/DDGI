from .dataset import PairDataset
from .utils import (
    prepare_adata,
    prepare_datasets,
    load_prior_data, 
    load_batchs,
    convert_newnames,
    agg_preds,
    reassemble_anndata,
    concat_results
)

__all__ = [
    'PairDataset',
    'prepare_adata',
    'prepare_datasets',
    'load_prior_data',
    'load_batchs',
    'convert_newnames',
    'agg_preds',
    'reassemble_anndata',
    'concat_results'
]

# test module
if __name__ == '__main__':
    import torch
    from driver_genes.data import prepare_adata, PairDataset
    from torch.utils.data import DataLoader
    # data config
    data_dir = '/users/s1155184323/projects/GraphDevPert/experiments/20250313-datasets/norman'
    data_file = 'preprocessed.h5ad'
    split_file = 'split_random.csv'
    sample_size = 200
    single_gene = False
    layer_name = 'X'
    var_subset = 'highly_variable'
    # load data
    adata, split_df, pertgenes, organism = prepare_adata(
        adata_path=f'{data_dir}/{data_file}',
        split_path=f'{data_dir}/{split_file}',
        single_gene=single_gene,
        layer_name=layer_name,
        var_subset='highly_variable'
    )
    dataset_noneg = PairDataset(
        adata=adata,
        pertgene_list=pertgenes,
        sample_size=sample_size,
        sample_neg=False
    )
    dataset_neg = PairDataset(
        adata=adata,
        pertgene_list=pertgenes,
        sample_size=sample_size,
        sample_neg=True
    )
    
    for X, Y, psi in DataLoader(dataset_noneg, batch_size=100):
        print(psi.sum(dim=1).min())
    
    for Xn, Yn, psin in DataLoader(dataset_neg, batch_size=100):
        print(psin.sum(dim=1).min())
    
    
