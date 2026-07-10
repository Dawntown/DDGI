from torchmetrics import MetricCollection
from torch import Tensor

from ..myutils import is_verbose
from .tmclass import (
    MyMultilabelAccuracy, MyMultilabelRankingLoss, MyMultilabelCoverageError, 
    MyMultilabelRelativePosition, MyMultilabelMRR, MyMultilabelNDCG, MyMultilabelAUROC, 
    MyMultilabelAUPRC, MyMultilabelRecall, MyMultilabelPrecision, MyMultilabelF1
)

def parse_metric_key(metric_key: str, ignore_const: bool = True, **kwargs):
    """Parse metric key to extract metric name and parameters.
    
    Args:
        metric_key: Metric key string (e.g., 'label#ACC@TopK', 'label#Recall@TopK')
        num_labels: Number of labels for the multilabel task
        num_samples: Number of samples in the dataset
    Returns:
        metric_name: Name of the metric (e.g., 'ACC', 'Recall')
        metric_params: Dictionary of parameters for the metric
    """
    label_wise = True if metric_key.startswith('label') else False
    topk = metric_key.split('@Top')[-1] if '@' in metric_key else None
    if isinstance(topk, str) and topk.isdigit():
        topk = int(topk)
        
    metric_obj = None
    metric_key_ = None
    if "ACC" in metric_key:
        metric_obj = MyMultilabelAccuracy(criteria='overlap', topk=topk, ignore_const=ignore_const)
        metric_key_ = f"ACC@Top{topk if isinstance(topk, int) else 'A'}"
    elif "Accuracy" in metric_key:
        metric_obj = MyMultilabelAccuracy(criteria='exact_match', threshold=0.5, ignore_const=False)
        metric_key_ = f"Accuracy"
    # natively sample-wise metrics should not ignore constant labels
    elif "RankingLoss" in metric_key:
        metric_obj = MyMultilabelRankingLoss(ignore_const=ignore_const)    
        metric_key_ = "RankingLoss"
    elif "Coverage" in metric_key:    
        metric_obj = MyMultilabelCoverageError(ignore_const=ignore_const)    
        metric_key_ = "Coverage"
    elif "RelPos" in metric_key:
        metric_obj = MyMultilabelRelativePosition(ignore_const=ignore_const)
        metric_key_ = "RelPos"
    elif "MRR" in metric_key:
        metric_obj = MyMultilabelMRR(ignore_const=ignore_const)
        metric_key_ = "MRR"
    elif "nDCG" in metric_key:
        metric_obj = MyMultilabelNDCG(ignore_const=ignore_const)
        metric_key_ = "nDCG"
    elif "AUROC" in metric_key:
        wise = 'label' if label_wise else 'sample'
        metric_obj = MyMultilabelAUROC(
            wise=wise, average='macro', 
            ignore_const=ignore_const
        )
        metric_key_ = f"{wise}#AUROC"
    elif "AUPRC" in metric_key:
        wise = 'label' if label_wise else 'sample'
        metric_obj = MyMultilabelAUPRC(
            wise=wise, average='macro', 
            ignore_const=ignore_const
        )
        metric_key_ = f"{wise}#AUPRC"
    elif "Recall" in metric_key: 
        # TODO: avoid label-wise TopK metrics 
        # they are not rational, whichever topk is ranked with sample or label is not the same.
        # In the sample-ranked topk will induce a very sparse prediction 
        # due to the number of positive sample on some label is uncertain.
        # in the label-ranked topk will is reasonable,
        # but if we ignore the constant labels, the topk will be very dense,
        # sometimes the topk will be all 1s, which is not reasonable.
        wise = 'label' if label_wise else 'sample'
        if wise == 'label':
            if topk is not None:
                Warning(f"Label-wise Recall with topk threshold is not rational regardless of ignore_const: too large if True else too small.")
                topk = None
            threshold = 0.5
        else:
            threshold = None
            
        metric_obj = MyMultilabelRecall(
            wise=wise, topk=topk, average='macro', 
            ignore_const=ignore_const, 
            threshold=threshold
        )
        metric_key_ = f"{wise}#Recall" + (f"@Top{topk}" if topk is not None else "")
        
    elif "Precision" in metric_key:
        wise = 'label' if label_wise else 'sample'
        
        if wise == 'label':
            if topk is not None:
                Warning(f"Label-wise Precision with topk threshold is not rational regardless of ignore_const: too large if True else too small.")
                topk = None
            threshold = 0.5
        else:
            threshold = None
            
        metric_obj = MyMultilabelPrecision(
            wise=wise, topk=topk, average='macro', 
            ignore_const=ignore_const,
            threshold=threshold,
        )
        metric_key_ = f"{wise}#Precision" + (f"@Top{topk}" if topk is not None else "")
    elif "F1" in metric_key:
        wise = 'label' if label_wise else 'sample'
        if wise == 'label':
            if topk is not None:
                Warning(f"Label-wise F1 with topk threshold is not rational regardless of ignore_const: too large if True else too small.")
                topk = None
            threshold = 0.5
        else:
            threshold = None
            
        metric_obj = MyMultilabelF1(
            wise=wise, topk=topk, average='macro', 
            ignore_const=ignore_const,
            threshold=threshold
        )
        metric_key_ = f"{wise}#F1" + (f"@Top{topk}" if topk is not None else "")
    
    #############################
    if metric_obj is None:
        raise ValueError(f"Invalid metric key: {metric_key}")
    
    if metric_key_ != metric_key:
        if is_verbose():
            print(f"Warning: Metric key {metric_key} was changed to {metric_key_}")
    return metric_key_, metric_obj



def fetch_metrics(
    metric_key_list: str | list[str],
    ignore_const: bool = True,
    **kwargs
    ):
    """Fetch a collection of metrics based on the provided list of metric names.
    
    Args:
        metrics: List of metric names to fetch with formats like 
            label#ACC@TopK, label#Recall@TopK, etc.
        ignore_const: Whether to ignore the constant labels.
        **kwargs: Additional keyword arguments for metric initialization.
        
    Note:
        label-wise metrics equals to sample ranking metrics and then average over all samples,
        sample-wise metrics equals to label ranking metrics and then average over all labels.
    
    Returns:
        MetricCollection: A collection of initialized metrics.
    """
    if metric_key_list == "core":
        metric_key_list = [
            "label#F1",
            "label#AUPRC",
            "label#AUROC",
            "ACC@Top1", # natively sample-wise
            "sample#Recall@Top5", # natively sample-wise
            "RelPos", # natively sample-wise
        ]
    elif metric_key_list == "all":
        metric_key_list = [
            "sample#AUPRC",
            "sample#AUROC",
            "label#AUROC",
            "label#AUPRC",
            "ACC@Top1", # natively sample-wise
            "ACC@Top5", # natively sample-wise
            "label#Recall", # using threshold 0.5
            "label#Precision", # using threshold 0.5
            "label#F1", # using threshold 0.5
            "sample#Recall@Top1",
            "sample#Recall@Top5",
            "sample#Precision@Top1",
            "sample#Precision@Top5",
            "sample#F1@Top1",
            "sample#F1@Top5",
            "RankingLoss", # natively sample-wise
            "Coverage", # natively sample-wise
            "RelPos", # natively sample-wise
            "MRR", # natively sample-wise
            "nDCG", # natively sample-wise
        ]
    elif metric_key_list == "core_samplelevel":
        # sample-level metrics means taking average over all labels
        metric_key_list = [
            "label#AUPRC",
            "label#AUROC",
            "label#F1",
            "label#Recall",
            "label#Precision",
            "Accuracy",
        ]
    elif metric_key_list == "core_labellevel":
        # label-level metrics means taking average over all samples
        metric_key_list = [
            "sample#AUPRC",
            "sample#AUROC",
            "sample#F1@Top1",
            "sample#F1@Top5",
            "sample#Recall@Top1",
            "sample#Recall@Top5",
            "sample#Precision@Top1",
            "sample#Precision@Top5",
        ]
    
    if isinstance(metric_key_list, str):
        metric_key_list = [metric_key_list]
        
        
    metric_dict = {}

    for metric_key in metric_key_list:
        metric_key_, metric_obj = parse_metric_key(metric_key, ignore_const=ignore_const)
        if metric_key_ in metric_dict:
            if is_verbose():
                print(f"Warning: Metric key {metric_key_}({metric_key}) already exists. Overwriting.")
        metric_dict[metric_key_] = metric_obj
    metric_collection = MetricCollection(metric_dict)
    return metric_collection


def reset_metrics(metrics_collection: MetricCollection | dict):
    if isinstance(metrics_collection, MetricCollection):
        metrics_collection.reset()
    elif isinstance(metrics_collection, dict):
        for k, v in metrics_collection.items():
            reset_metrics(v)
            
        
def get_scalar_metrics_dict(metrics_dict: dict):
    return {k: v.item() for k, v in metrics_dict.items() if isinstance(v, Tensor) and v.numel() == 1}

####################
# Test all:
if __name__ == '__main__':
    # test
    import numpy as np
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    torch.manual_seed(1)
    num_labels = 105
    num_samples = 1024
    k = 10
    preds1 = torch.tensor(np.random.normal(size=(num_samples, num_labels)))
    preds2 = torch.tensor(np.random.normal(size=(num_samples, num_labels)))
    target = ((preds1 + 0.7 * torch.tensor(np.random.normal(size=(num_samples, num_labels)))) > 1).float()

    # Fetch all metrics
    # metrics = fetch_metrics("all", num_samples, num_labels)

    # for name, metric in metrics.items():
    #         # Good performance
    #         metric.update(preds1, target)
    #         good_value = metric.compute()
    #         print(f"{name}:")
    #         print(f"\tGood performance: {good_value}")
    #         metric.reset()
            
    #         # Bad performance
    #         metric.update(preds2, target)
    #         bad_value = metric.compute()
    #         print(f"\tBad performance: {bad_value}")
    #         metric.reset()

    #         print()
    
    sparse_target_dict = {}
    for k in [1, 10, 20, 50, 80, 105]:
        sparse_target_dict[f't{k:03d}'] = torch.concat((target[:,:k], torch.zeros_like(target)[:,k:]), dim=1)


    
    metric_funcs = fetch_metrics("all", ignore_const=False)
    metric_funcs_ign = fetch_metrics("all", ignore_const=True)
    
    result_list = []
    for tn, tt in sparse_target_dict.items():
        good_dict = get_scalar_metrics_dict(metric_funcs(preds1, tt))
        good_dict.update({'sparse': tn, 'performance': 'good', 'ignore': False})
        result_list.append(good_dict)
        good_dict_ign = get_scalar_metrics_dict(metric_funcs_ign(preds1, tt))
        good_dict_ign.update({'sparse': tn, 'performance': 'good', 'ignore': True})
        result_list.append(good_dict_ign)
        metric_funcs.reset()
        metric_funcs_ign.reset()

        bad_dict = get_scalar_metrics_dict(metric_funcs(preds2, tt))
        bad_dict.update({'sparse': tn, 'performance': 'bad', 'ignore': False})
        result_list.append(bad_dict)
        bad_dict_ign = get_scalar_metrics_dict(metric_funcs_ign(preds2, tt))
        bad_dict_ign.update({'sparse': tn, 'performance': 'bad', 'ignore': True})
        result_list.append(bad_dict_ign)
        metric_funcs.reset()
        metric_funcs_ign.reset()
        
    df = pd.DataFrame(result_list)
    sel_metrics = [
        'label#AUROC', 'label#AUPRC',
        'sample#AUROC', 'sample#AUPRC', 
        'ACC@Top1', 'ACC@Top5',
        'RelPos', 'RankingLoss',
    ]
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    for i, metric in enumerate(sel_metrics):
        ax = axes[i//2, i%2]
        sns.lineplot(
            data=df,
            x='sparse',
            y=metric,
            hue='performance',
            style='ignore',
            ax=ax,
        )
        if i != 3:
            ax.legend().remove()
        else:
            ax.legend(
                loc='upper left', 
                bbox_to_anchor=(1.05, 1.0),
                frameon=False
            )
    plt.tight_layout()