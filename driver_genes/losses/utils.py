import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from ..myutils import Args, args_to_dict, is_verbose

class UncertaintyWeighting(nn.Module):
    """Uncertainty Weighting for multi-task learning.
    
    This method learns the weights of different tasks by modeling the uncertainty
    of each task. The weights are parameterized as learnable parameters.
    
    Reference:
        Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty 
        to weigh losses for scene geometry and semantics. CVPR.
    """
    def __init__(self, init_weights: Optional[Union[Dict[str, float], Args]]):
        """Initialize the uncertainty weighting module.
        
        Args:
            init_weights: Dictionary of initial weights for each task. If None, all weights are initialized to 1.0
        """
        super().__init__()
        init_weights = args_to_dict(init_weights)        
        # Initialize log variances (weights) as learnable parameters
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(-torch.log(torch.tensor(w)))
            for name, w in init_weights.items()
        })
        if is_verbose():
            print("Using uncertainty loss weighting strategy with initial weights: ", init_weights)
        
    def forward(self, losses: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """Compute the weighted loss.
        
        Args:
            losses: Dictionary of loss values for each task
            
        Returns:
            Tuple containing:
                - Total weighted loss
                - Dictionary of individual weighted losses and weights
        """
        
        # Compute weights from log variances
        weights = {name: torch.exp(-self.log_vars[name]) for name in self.log_vars.keys() if name in losses.keys()}
        
        # weight_sum = sum(weights.values())
        # weights = {name: w / weight_sum for name, w in weights.items()}
        
        # Compute weighted losses
        weighted_losses = {}
        for name in weights.keys():
            weighted_loss = weights[name] * losses.get(name, torch.tensor(0.0)) + self.log_vars[name]
            weighted_losses[name] = weighted_loss
            
        total_loss = sum(weighted_losses.values())
        
        # Create dictionary of results
        results = {'mix': total_loss.detach()}
        results.update({f'{name}_w': w.detach() for name, w in weights.items()})
        results.update({name: l.detach() for name, l in losses.items()})
        
        return total_loss, results


class PredefinedWeightLoss:
    """Predefined weight loss mixing strategy"""
    def __init__(self, weights: Union[Dict[str, float], Args]):
        """
        Args:
            weights: Dictionary of predefined weights for each task
        """
        self.weights = args_to_dict(weights)
        if is_verbose():
            print("Using predefined weight loss with weights: ", self.weights)
        
    def __call__(self, losses: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute weighted loss
        
        Args:
            losses: Dictionary of loss values for each task
            
        Returns:
            Tuple containing:
                - Total weighted loss
                - Dictionary of individual weighted losses
        """
        # Use task name as key to lookup corresponding loss (original implementation used weight values as keys)
        weighted_losses = {
            task_name: weight * losses.get(task_name, torch.tensor(0.0))
            for task_name, weight in self.weights.items()
        }
        total_loss = sum(weighted_losses.values())
        
        results = {'mix': total_loss.detach()}
        results.update({name: l.detach() for name, l in losses.items()})
        
        return total_loss, results


class AdaptiveWeightLoss:
    """Adaptive weight loss mixing strategy"""
    def __init__(self, task_names: List[str] = None):
        self.task_names = task_names
        
    def __call__(self, losses: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute adaptive weighted loss
        
        Args:
            losses: Dictionary of loss values for each task
            
        Returns:
            Tuple containing:
                - Total weighted loss
                - Dictionary of individual weighted losses
        """
        weights = {k: losses['rec'].detach() / losses.get(k, torch.inf).detach() for k in self.task_names}
        weighted_losses = {k: losses.get(k, torch.tensor(0.0)) * weights[k] for k in self.task_names}
        total_loss = sum(weighted_losses.values())
        
        results = {'mix': total_loss.detach()}
        results.update({f'{name}_w': weights[name].detach() for name in self.task_names})
        results.update({name: l.detach() for name, l in losses.items()})
        
        return total_loss, results

