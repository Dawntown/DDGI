import torch
from torch import Tensor
from typing import Tuple
from torchmetrics import Metric
from torchmetrics.functional.classification import (
    multilabel_auroc, multilabel_average_precision, 
    multilabel_recall, multilabel_precision, multilabel_f1_score,
    multilabel_ranking_average_precision
)
from .utils import (
    _filter_labels, _filter_samples,
    _thresholding, _ranking
)


class MyMultilabelCounter(Metric):
    """Counter for predictions with probability > threshold per sample.
    Args:
        threshold: Probability threshold for positive predictions
        dist_sync_on_step: Synchronize metric state between processes at each step
    """
    
    def __init__(self,threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        
        # Add states for counting
        self.add_state("counts", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, *args) -> None:
        """Update state with predictions.
        
        Args:
            preds: Predictions tensor with shape (N, C) or (N, C, ...) 
                  where N is batch size and C is number of classes/labels
        """
            
        # Count predictions above threshold per sample
        pred_counts = (preds > self.threshold).sum(dim=1).float()
        
        # Update states
        self.counts += pred_counts.sum()
        self.total += preds.shape[0]
        
    def compute(self) -> torch.Tensor:
        """Compute average number of positive predictions per sample."""
        return self.counts / self.total




class MyMultilabelAccuracy(Metric):
    def __init__(self, 
                 criteria: str = 'exact_match', 
                 topk: str | int = None, 
                 threshold: float = None,
                 ignore_const: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        valid_criteria = ['exact_match', 'hamming', 'overlap', 'contain', 'belong', 'jaccard']
        if criteria not in valid_criteria:
            raise ValueError(f"criteria must be one of {valid_criteria}")
        
        self.criteria = criteria
        self.topk = topk
        self.threshold = threshold
        self.ignore_const = ignore_const
        
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:

        if self.ignore_const:
            preds, target = _filter_samples(preds, target)
            
        preds = _thresholding(preds, target, self.topk, threshold=self.threshold)
        if self.criteria == 'exact_match':
            correct = (preds == target).all(dim=1)
        elif self.criteria == 'hamming':
            correct = (preds == target).float().mean(dim=1)
        elif self.criteria == 'overlap':
            correct = ((preds == 1) & (target == 1)).any(dim=1)
        elif self.criteria == 'contain':
            # All positive labels in target are predicted as positive
            correct = ((preds == 1) >= (target == 1)).all(dim=1)
        elif self.criteria == 'jaccard':
            correct = ((preds == 1) & (target == 1)).sum(dim=1) / ((preds == 1) | (target == 1)).sum(dim=1)
        elif self.criteria == 'belong':  # belong
            # All positive predictions are true positive labels
            correct = ((preds == 1) <= (target == 1)).all(dim=1)
        else:
            raise ValueError(f"Invalid criteria: {self.criteria}")
        
        self.correct += correct.sum()
        self.total += target.size(0)

    def compute(self) -> Tensor:
        return self.correct / self.total



class MyMultilabelRelativePosition(Metric):
    def __init__(self, ignore_const: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_const = ignore_const
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        # ranking = preds.argsort(dim=1, descending=True).argsort(dim=1, descending=False).float() + 1
        # the ranking of the target label in the preds
        # ranking = 1 - (ranking / preds.size(1))
        # tensor.topk is faster than tensor.argsort

        if self.ignore_const:
            preds, target = _filter_samples(preds, target)
            
        ranking = _ranking(preds, normalized=True)        
        correct = (ranking * target).sum(dim=1) / target.sum(dim=1)
        
        self.correct += correct.sum()
        self.total += target.size(0)
        
    def compute(self) -> Tensor:
        return self.correct / self.total

    

class MyMultilabelRankingLoss(Metric):
    """
    Custom implementation of Multilabel Ranking Loss.
    Measures the average number of pairs of incorrect labels ranked higher
    than true labels.
    """
    def __init__(self, ignore_const: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_const = ignore_const
        self.add_state("ranking_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        
        if self.ignore_const:
            preds, target = _filter_samples(preds, target)
            
        loss_sum, n_samples = self._multilabel_ranking_loss_update(preds, target)
        self.ranking_loss += loss_sum
        self.total += n_samples
        
    def compute(self) -> Tensor:
        return self.ranking_loss / self.total
    
    @staticmethod
    def _multilabel_ranking_loss_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
        """Accumulate state for label ranking loss. (from torchmetrics)

        Args:
            preds: tensor with predictions
            target: tensor with ground truth labels

        """
        num_preds, num_labels = preds.shape
        relevant = target == 1
        num_relevant = relevant.sum(dim=1)

        # Ignore instances where number of true labels is 0 or n_labels
        mask = (num_relevant > 0) & (num_relevant < num_labels)
        preds = preds[mask]
        relevant = relevant[mask]
        num_relevant = num_relevant[mask]

        # Nothing is relevant
        if len(preds) == 0:
            return torch.tensor(0.0, device=preds.device), 1

        inverse = preds.argsort(dim=1).argsort(dim=1)
        per_label_loss = ((num_labels - inverse) * relevant).to(torch.float32)
        correction = 0.5 * num_relevant * (num_relevant + 1)
        denom = num_relevant * (num_labels - num_relevant)
        loss = (per_label_loss.sum(dim=1) - correction) / denom
        return loss.sum(), num_preds



class MyMultilabelCoverageError(Metric):
    """
    Custom implementation of Multilabel Coverage Error.
    Measures the average number of labels that need to be included in the 
    prediction to cover all true labels.
    """
    def __init__(self, ignore_const: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_const = ignore_const
        self.add_state("coverage_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        
        if self.ignore_const:
            preds, target = _filter_samples(preds, target)
            
        coverages_sum, n_samples = self._multilabel_coverage_error_update(preds, target)
        
        self.coverage_sum += coverages_sum
        self.total += n_samples
        
    def compute(self) -> Tensor:
        return self.coverage_sum / self.total
    
    @staticmethod
    def _multilabel_coverage_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
        """Accumulate state for coverage error. (from torchmetrics)"""        
        offset = torch.zeros_like(preds)
        offset[target == 0] = preds.min().abs() + 10  # Any number >1 works
        preds_mod = preds + offset
        preds_min = preds_mod.min(dim=1)[0]
        coverage = (preds >= preds_min[:, None]).sum(dim=1).to(torch.float32)
        return coverage.sum(), coverage.numel()
        

    
class MyMultilabelMRR(Metric):
    def __init__(self, ignore_const: bool = True, **kwargs):
        super().__init__(**kwargs)
        # Define state variables: 
        # mrr_sum for accumulating the sum of reciprocal ranks, 
        # total for tracking the number of samples
        # here, we only implement the average MRR
        self.ignore_const = ignore_const
        self.add_state("mrr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        
        if self.ignore_const:
            preds, target = _filter_samples(preds, target)
            
        reciprocal_ranks_sum, n_samples = self._retrieval_reciprocal_rank(preds, target)
        
        self.mrr_sum += reciprocal_ranks_sum
        self.total += n_samples
        
    def compute(self) -> Tensor:
        # Compute the average MRR
        return self.mrr_sum / self.total
    
    @staticmethod
    def _retrieval_reciprocal_rank(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
        """Compute reciprocal rank (for information retrieval) (customed)
        It is actually the inversed top-1 ranked true label position.
        """
        ranking = _ranking(preds, normalized=False)
        offset = torch.zeros_like(ranking)
        offset[target == 0] = preds.size(-1)
        ranking = ranking + offset
        ranking_top1true = ranking.min(dim=-1).values.float()
        ranking_top1true[torch.nonzero(target.sum(dim=-1) == 0)] = torch.inf
        return (1 / (ranking_top1true + 1.0)).sum(), preds.size(0)



class MyMultilabelNDCG(Metric):
    def __init__(self, ignore_const: bool = True, **kwargs):
        super().__init__(**kwargs)
        # Define state variables: ndcg_sum for accumulating nDCG values, total for tracking the number of samples
        self.ignore_const = ignore_const
        self.add_state("ndcg_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        
        if self.ignore_const:
            preds, target = _filter_samples(preds, target)
            
        ndcg_sum, num_samples = self._retrieval_ndcg(preds, target)
        self.ndcg_sum += ndcg_sum
        self.total += num_samples
        
    def compute(self) -> Tensor:
        # Compute the average nDCG
        return self.ndcg_sum / self.total
    
    @staticmethod
    def _retrieval_ndcg(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
        """Compute `Normalized Discounted Cumulative Gain`_ (for information retrieval)
        customed by Xiaoran
        """
        # Skip samples with no positive labels
        valid_samples = target.sum(dim=1) > 0
        if not valid_samples.any():
            return
        
        preds = preds[valid_samples]
        target = target[valid_samples]
        n_samples = preds.size(0)
        
        # Get the ranking indices for all samples based on predictions
        rankings = preds.argsort(dim=1, descending=True)
        
        # Compute DCG for all samples
        ranked_scores = target.gather(1, rankings)
        positions = torch.arange(2, preds.size(1) + 2, device=preds.device).float()
        discount = torch.log2(positions)  # log2(2), log2(3), ..., log2(n+1)
        dcg = (ranked_scores / discount).sum(dim=1)
        
        # Compute IDCG for all samples
        ideal_scores = target.sort(dim=1, descending=True)[0]
        idcg = (ideal_scores / discount).sum(dim=1)
        
        # Calculate nDCG and update state
        ndcg = dcg / idcg.clamp(min=1e-10)  # Avoid division by zero
        return ndcg.sum(), n_samples
    

class MyMultilabelAUROC(Metric):
    def __init__(self, 
                 wise: str = 'label', 
                 average: str = 'macro', 
                 ignore_const: bool = True,
                 **kwargs):
        """
        Initialize the multilabel AUROC metric. 
        We only implement the AUROC within batch due to large number of smaple pairs
        
        Args:
            wise (str): Computation type, 'label' for label-wise or 'sample' for sample-wise.
        """
        super().__init__(**kwargs)
        assert wise in ['label', 'sample'], "wise must be either 'label' or 'sample'"
        assert average in ['micro', 'macro'], "average must be either 'micro' or 'macro'"
        
        self.wise = wise  # 'label' or 'sample'
        self.average = average # micro, macro #, none
        self.ignore_const = ignore_const
        self.add_state("auc_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the state with new predictions and targets.
        
        Args:
            preds (Tensor): Predicted probabilities, shape (n_samples, n_labels)
            target (Tensor): True labels, shape (n_samples, n_labels)
        """
        if self.ignore_const and self.wise == 'label':
            preds, target = _filter_labels(preds, target)
            
        if self.ignore_const and self.wise == 'sample':
            preds, target = _filter_samples(preds, target)
            
        if self.wise == 'label':
            # Label-wise AUROC
            num_labels = preds.size(1)  # Dynamically get number of labels
            auc = multilabel_auroc(
                preds, target, 
                num_labels=num_labels, 
                average=self.average,
                validate_args=False,
            )
        else:
            # Sample-wise AUROC
            num_samples = preds.size(0)  # Dynamically get number of samples
            auc = multilabel_auroc(
                preds.T, target.T,
                num_labels=num_samples,
                average=self.average,
                validate_args=False,
            )
                
        self.auc_sum += auc
        self.total += 1
        
    def compute(self) -> Tensor:
        """
        Compute the final average AUROC.
        
        Returns:
            Tensor: Average AUROC value
        """
        return self.auc_sum / self.total
    

class MyMultilabelAUPRC(Metric):
    def __init__(self, wise='label', average='macro', ignore_const: bool = True, **kwargs):
        """
        Initialize the multilabel AUPRC metric.
        
        Args:
            wise (str): Computation type, 'label' for label-wise or 'sample' for sample-wise.
        """
        super().__init__(**kwargs)
        assert wise in ['label', 'sample'], "wise must be either 'label' or 'sample'"
        # assert average in ['micro', 'macro'], "average must be either 'micro' or 'macro'"
        self.wise = wise  # 'label' or 'sample'
        # self.average = average # micro, macro #, none
        self.ignore_const = ignore_const
        
        self.add_state("auprc_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the state with new predictions and targets.
        
        Args:
            preds (Tensor): Predicted probabilities, shape (n_samples, n_labels)
            target (Tensor): True labels, shape (n_samples, n_labels)
        """
        if self.ignore_const and self.wise == 'label':
            preds, target = _filter_labels(preds, target)
            
        if self.ignore_const and self.wise == 'sample':
            preds, target = _filter_samples(preds, target)
            
        if self.wise == 'label':
            # Label-wise AUPRC, i.e. average precision
            num_samples = preds.size(0)
            auprc = multilabel_ranking_average_precision(
                # implementation of LRAP is more robust and faster than AP
                # lrap(preds, target) == ap(preds.T, target.T)
                preds.T, target.T,
                num_labels=num_samples,
                validate_args=False,
            )
        else:
            # Sample-wise AUPRC, i.e. label ranking average precision
            num_labels = preds.size(1)
            auprc = multilabel_ranking_average_precision(
                preds, target,
                num_labels=num_labels,
                validate_args=False,
            )
            
        self.auprc_sum += auprc
        self.total += 1
            
        
    def compute(self) -> Tensor:
        """
        Compute the final average AUPRC.
        
        Returns:
            Tensor: Average AUPRC value
        """
        return self.auprc_sum / self.total
    

class MyMultilabelRecall(Metric):
    def __init__(self, 
                 wise: str = 'label', 
                 topk: int | str = None, 
                 threshold: float = None, 
                 average: str = 'macro',
                 ignore_const: bool = True,
                 **kwargs):
        """
        Initialize the multilabel Recall metric.
        
        Args:
            wise (str): 'label' for label-wise or 'sample' for sample-wise computation.
            topk (int, optional): If specified, compute top-k Recall.
            **kwargs: Additional arguments for the Metric base class.
        """
        super().__init__(**kwargs)
        assert wise in ['label', 'sample'], "wise must be either 'label' or 'sample'"
        assert average in ['micro', 'macro'], "average must be either 'micro' or 'macro'"
        
        self.wise = wise
        self.topk = topk
        self.threshold = threshold
        self.average = average
        self.ignore_const = ignore_const
        
        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric state with new predictions and targets.
        
        Args:
            preds (Tensor): Predicted probabilities/scores, shape (n_samples, n_labels).
            target (Tensor): True labels, shape (n_samples, n_labels).
        """
        if self.ignore_const and self.wise == 'label':
            preds, target = _filter_labels(preds, target)
            
        if self.ignore_const and self.wise == 'sample':
            preds, target = _filter_samples(preds, target)
            
        preds = _thresholding(preds, target, self.topk, threshold=self.threshold)
        
        if self.wise == 'label':
            # Label-wise: Compute Recall per label and average
            num_labels = preds.size(1)
            recall = multilabel_recall(
                preds, target,
                num_labels=num_labels,
                average=self.average,
                validate_args=False,
            )
        else:
            # Sample-wise: Compute Recall per sample and average
            num_samples = preds.size(0)
            recall = multilabel_recall(
                preds.T, target.T,
                num_labels=num_samples,
                average=self.average,
                validate_args=False,
            )
        self.recall_sum += recall
        self.total += 1
    
    def compute(self) -> Tensor:
        """
        Compute the final average Recall.
        
        Returns:
            Tensor: Average Recall value.
        """
        return self.recall_sum / self.total
    


class MyMultilabelPrecision(Metric):
    def __init__(self, 
                 wise: str = 'label', 
                 topk: int | str = None, 
                 threshold: float = None,
                 average: str = 'macro',
                 ignore_const: bool = True,
                 **kwargs):
        """
        Initialize the multilabel Precision metric.
        
        Args:
            wise (str): 'label' for label-wise or 'sample' for sample-wise computation.
            topk (int, optional): If specified, compute top-k Precision.
            **kwargs: Additional arguments for the Metric base class.
        """
        super().__init__(**kwargs)
        assert wise in ['label', 'sample'], "wise must be either 'label' or 'sample'"
        assert average in ['micro', 'macro'], "average must be either 'micro' or 'macro'"
        
        self.wise = wise
        self.topk = topk
        self.threshold = threshold
        self.average = average
        self.ignore_const = ignore_const
        
        self.add_state("precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric state with new predictions and targets.
        
        Args:
            preds (Tensor): Predicted probabilities/scores, shape (n_samples, n_labels).
            target (Tensor): True labels, shape (n_samples, n_labels).
        """
        if self.ignore_const and self.wise == 'label':
            preds, target = _filter_labels(preds, target)
            
        if self.ignore_const and self.wise == 'sample':
            preds, target = _filter_samples(preds, target)
            
        preds = _thresholding(preds, target, self.topk, threshold=self.threshold)
        
        if self.wise == 'label':
            # Label-wise: Compute Precision per label and average
            num_labels = preds.size(1)
            precision = multilabel_precision(
                preds, target,
                num_labels=num_labels,
                average=self.average,
                validate_args=False,
            )
        else:
            # Sample-wise: Compute Precision per sample and average
            num_samples = preds.size(0)
            precision = multilabel_precision(
                preds.T, target.T,
                num_labels=num_samples,
                average=self.average,
                validate_args=False,
            )
        self.precision_sum += precision
        self.total += 1
    
    def compute(self) -> Tensor:
        """
        Compute the final average Precision.
        
        Returns:
            Tensor: Average Precision value.
        """
        return self.precision_sum / self.total
    
    

class MyMultilabelF1(Metric):
    def __init__(self, 
                 wise: str = 'label', 
                 topk: int | str = None, 
                 threshold: float = None,
                 average: str = 'macro',
                 ignore_const: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        assert wise in ['label', 'sample'], "wise must be either 'label' or 'sample'"
        assert average in ['micro', 'macro'], "average must be either 'micro' or 'macro'"
        
        self.wise = wise
        self.topk = topk
        self.threshold = threshold
        self.average = average
        self.ignore_const = ignore_const
        
        self.add_state("f1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric state with new predictions and targets.
        
        Args:
            preds (Tensor): Predicted probabilities/scores, shape (n_samples, n_labels).
            target (Tensor): True labels, shape (n_samples, n_labels).
        """
        if self.ignore_const and self.wise == 'label':
            preds, target = _filter_labels(preds, target)
            
        if self.ignore_const and self.wise == 'sample':
            preds, target = _filter_samples(preds, target)
            
            
        preds = _thresholding(preds, target, self.topk, threshold=self.threshold)
        
        if self.wise == 'label':
            # Label-wise: Compute F1 per label and average
            num_labels = preds.size(1)
            f1 = multilabel_f1_score(
                preds, target,
                num_labels=num_labels,
                average=self.average,
                validate_args=False,
            )
        else:
            # Sample-wise: Compute F1 per sample and average
            num_samples = preds.size(0)
            f1 = multilabel_f1_score(
                preds.T, target.T,
                num_labels=num_samples,
                average=self.average,
                validate_args=False,
            )
        self.f1_sum += f1
        self.total += 1
    
    def compute(self) -> Tensor:
        """
        Compute the final average F1.
        
        Returns:
            Tensor: Average F1 value.
        """
        return self.f1_sum / self.total

