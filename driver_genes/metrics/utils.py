import torch
from torch import Tensor
from typing import Tuple

def _thresholding(preds: Tensor, target: Tensor = None, topk: int | str = None, **kwargs) -> Tensor:
    """Threshold the predictions to only keep the top-k predictions.
    
    Args:
        preds: Predictions tensor with shape (N, C) where N is batch size and C is number of classes/labels
        target: True labels tensor with shape (N, C) (optional)
        topk: Number of top predictions to keep
        **kwargs: Additional keyword arguments for filtering
    
    Returns:
        Tensor: Filtered predictions tensor with shape (N, C)
    """
    
    if target is not None:
        assert preds.shape == target.shape, "Predictions and targets must have the same dims"
    
    if topk is not None:
        # tensor.topk is faster than tensor.sort and tensor.argsort in either cpu and gpu
        if isinstance(topk, int):
            k = min(topk, preds.size(-1))  # Ensure k doesn’t exceed number of labels
            threshold = preds.topk(k=k, dim=-1, sorted=True).values[:, -1].unsqueeze(dim=-1)
        elif topk == 'adaptive' or topk == 'A':
            # print("Adaptive topk")
            threshold = preds.topk(k=preds.size(-1), dim=-1, largest=True).values[
                range(preds.size(0)), 
                target.sum(dim=-1)-1
            ].unsqueeze(dim=-1)
            threshold[torch.nonzero(target.sum(dim=-1) == 0)] = torch.inf
        else:
            raise ValueError(f"Invalid top_k: {topk}")
        
    else:
        threshold = kwargs.get('threshold', 0.5)  # Default threshold for binary classification
        # to stablize the estimated probabilities, we use MinMax scaling
        preds = (preds - preds.min()) / (preds.max() - preds.min())
    
    return (preds >= threshold).float()

def _filter_labels(preds: Tensor, target: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    """Filter out the labels without any positive sample.
    """
    kept_labels = target.detach().sum(dim=0) > 0
    return preds[:, kept_labels], target[:, kept_labels]

def _filter_samples(preds: Tensor, target: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    """Filter out the samples without any positive label.
    """
    
    kept_samples = target.detach().sum(dim=1) > 0
    # updated the module and loss for stable training process
    return preds[kept_samples], target[kept_samples]


def _ranking(preds: Tensor, normalized: bool = False) -> Tensor:
    """Rank the predictions in descending order.
    
    Args:
        preds: Predictions tensor with shape (N, C) where N is batch size and C is number of classes/labels
    
    Returns:
        Tensor: Ranked predictions tensor with shape (N, C)
    """
    
    # Get the indices that would sort the tensor
    if normalized == False:
        ranking = preds.topk(k=preds.size(-1), dim=-1, largest=True).indices
        ranking = ranking.topk(k=preds.size(-1), dim=-1, largest=False).indices
        return ranking
    else:
        # Get the indices that would sort the tensor
        ranking = preds.topk(k=preds.size(-1), dim=-1, largest=False).indices
        ranking = ranking.topk(k=preds.size(-1), dim=-1, largest=False).indices
        # Normalize the ranking to [0, 1]
        ranking = ranking / (preds.size(-1) - 1)
        return ranking

