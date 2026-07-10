import torch
from torch import Tensor

def edist_euclidean(X: Tensor, Y: Tensor) -> Tensor:
    delta_xy = torch.cdist(X, Y).mean() # euclidean distance
    sigma_x = torch.cdist(X, X).mean()
    sigma_y = torch.cdist(Y, Y).mean()
    return 2 * delta_xy - sigma_x - sigma_y

def edist_squared_euclidean(X: Tensor, Y: Tensor) -> Tensor:
    delta_xy = torch.cdist(X, Y).pow(2).mean() # squared euclidean distance
    sigma_x = torch.cdist(X, X).pow(2).mean()
    sigma_y = torch.cdist(Y, Y).pow(2).mean()
    return 2 * delta_xy - sigma_x - sigma_y

def mmdist_linear(X: Tensor, Y: Tensor) -> Tensor:
    kernel_xx = torch.mm(X, X.t())
    kernel_yy = torch.mm(Y, Y.t())
    kernel_xy = torch.mm(X, Y.t())
    return kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()

def mmdist_gaussian(X: Tensor, Y: Tensor, sigma: float = 1.0) -> Tensor:
    kernel_xx = gaussian_kernel(X, X, sigma)
    kernel_yy = gaussian_kernel(Y, Y, sigma)
    kernel_xy = gaussian_kernel(X, Y, sigma)
    return kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()


def gaussian_kernel(x: Tensor, y: Tensor, sigma: float = 1.0) -> Tensor:
    """
    Computes the Gaussian (RBF) kernel between two sets of samples.
    """
    x_expanded = x.unsqueeze(1)  # Shape: (batch_size_x, 1, feature_dim)
    y_expanded = y.unsqueeze(0)  # Shape: (1, batch_size_y, feature_dim)
    return torch.exp(-((x_expanded - y_expanded) ** 2).sum(dim=2) / (2 * sigma**2))

def kl_div(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()