import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


## losses of reconstruction
def mse(X: Tensor, Y: Tensor) -> Tensor:
    return F.mse_loss(X, Y, reduction='none').mean() * 1000 # for comparable across datasets


def zinb(X: Tensor, params: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """
    ZINB loss function for reconstruction loss of count data
    X: (n_cells, n_genes) - observed counts
    params: mean, dispersion, pi - model parameters
        mu: expected value of the negative binomial (0, infinity)
        theta: dispersion parameter of the negative binomial (0, infinity)
        pilogit: dropout logit (zero inflation) (-infinity, infinity)
    """
    mu, theta, pilogit = params
    eps = 1e-8
        
    neg_log_pi = F.softplus(-pilogit)
    log_theta_mu = torch.log(theta + mu + eps)
    neg_pilogit_theta_log = -pilogit + theta * (torch.log(theta + eps) - log_theta_mu)
    
    case_zero = F.softplus(neg_pilogit_theta_log) - neg_log_pi
    case_nonzero = (
        -neg_log_pi
        + neg_pilogit_theta_log
        + X * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(X + theta) 
        - torch.lgamma(theta) 
        - torch.lgamma(X + 1)
    )
    
    # Compute zero-inflated log probability
    log_prob = torch.where(
        X < eps,
        case_zero,
        case_nonzero
    )
    
    # Return negative log likelihood
    return -log_prob.mean() * 1000

