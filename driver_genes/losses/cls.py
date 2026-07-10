import torch
import torch.nn.functional as F
from torch import Tensor


from ..metrics.utils import _filter_labels

## losses of classification
def binary_cross_entropy(cls: Tensor, psi: Tensor, ignore_const: bool = True) -> Tensor:
    if ignore_const:
        cls, psi = _filter_labels(cls, psi)
    return F.binary_cross_entropy_with_logits(cls, psi.float(), reduction='none').mean() * 1000 # for comparable across datasets
    


def binary_cross_entropy_weighted(cls: Tensor, psi: Tensor, **kwargs) -> Tensor:
    alpha = (psi == 0).sum(dim=0) / psi.shape[0]
    weights = torch.where(psi == 1, alpha, 1 - alpha) + 1e-1
    return F.binary_cross_entropy_with_logits(cls, psi.float(), weight=weights, reduction='none').mean() * 1000  # for comparable across datasets


def focal_loss(cls: Tensor, psi: Tensor, alpha: float = None, gamma: float = 0.0, **kwargs) -> Tensor:
    if alpha is None:
        alpha = (psi == 0).sum(dim=0) / psi.shape[0]
    probs = torch.sigmoid(cls)
    pt = torch.where(psi == 1, probs, 1 - probs)
    focal_weight = (1 - pt).pow(gamma) + 1e-1
    alpha_weight = torch.where(psi == 1, alpha, 1 - alpha) + 1e-1
    ce_loss = F.binary_cross_entropy_with_logits(cls, psi.float(), reduction='none')
    focal_loss = alpha_weight * focal_weight * ce_loss
    return focal_loss.mean() * 1000 # for comparable across datasets


def multilabel_cross_entropy(logits: Tensor, labels: Tensor, inf: float = 1e12, **kwargs) -> Tensor:
    logits, labels = logits.float(), labels.float()
    y_pred = (1 - 2 * labels) * logits
    y_pred_neg = y_pred - labels * inf
    y_pred_pos = y_pred - (1 - labels) * inf
    zeros = torch.zeros_like(logits[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean() * 1000 # for comparable across datasets