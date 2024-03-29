##########################################
# Define kernel functions
##########################################

import numpy as np
import torch
from torch.distributions.normal import Normal
from scipy.stats import truncnorm

from .utils.utils import check_tensor


def raised_cosine_kernel(t, alpha, u, sigma):
    """Compute the raised cosine distribution kernel.

    Parameters
    ----------
    t : tensor | array-like
        timepoints to compute kernel value at

    alpha : 1d array like

    u : 1d array like
        u = m - sigma

    sigma : 1d array like

    Returns
    -------
    tensor
    """

    t = check_tensor(t)

    alpha = check_tensor(alpha)
    u = check_tensor(u)
    sigma = check_tensor(sigma)

    n_drivers = u.shape[0]
    kernels = []
    for i in range(n_drivers):
        kernel = (1 + torch.cos((t - u[i]) / sigma[i] * np.pi - np.pi)) \
            / (2 * sigma[i])
        mask_kernel = (t < u[i]) | (t > (u[i] + 2*sigma[i]))
        kernel[mask_kernel] = 0.
        kernel *= alpha[i]
        kernels.append(kernel)

    return torch.stack(kernels, 0).float()


def truncated_gaussian_kernel(t, alpha, m, sigma, lower, upper, dt):
    """Compute the truncated normal distribution kernel.

    Parameters
    ----------
    t : tensor | array-like
        timepoints to compute kernel value at

    params : tensor | tuple
        model parameters (baseline, alpha, mu, sigma)

    lower, upper : floats
        truncation values of kernel's support

    Returns
    -------
    tensor
    """

    t = check_tensor(t)
    alpha = check_tensor(alpha)
    m = check_tensor(m)
    sigma = check_tensor(sigma)

    n_drivers = m.shape[0]
    kernels = []
    norm_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    for i in range(n_drivers):
        kernel = torch.exp((- torch.square(t - m[i]) /
                            (2 * torch.square(sigma[i]))))
        kernel = kernel + 0
        mask_kernel = (t < lower) | (t > upper)
        kernel[mask_kernel] = 0.
        C = (norm_dist.cdf((torch.tensor(upper)-m[i])/sigma[i]) -
             norm_dist.cdf((torch.tensor(lower)-m[i])/sigma[i])) * sigma[i] * np.sqrt(2*np.pi)
        # kernel /= (kernel.sum() * dt)
        kernel /= C
        kernel *= alpha[i]
        kernels.append(kernel)

    return torch.stack(kernels, 0).float()


def exponential_kernel(t, alpha, gamma, lower, upper, dt):
    """Compute the truncated normal distribution kernel.

    Parameters
    ----------
    t : tensor | array-like
        timepoints to compute kernel value at

    params : tensor | tuple
        model parameters (baseline, alpha, gamma)

    lower, upper : floats
        truncation values of kernel's support

    Returns
    -------
    tensor
    """

    t = check_tensor(t)
    alpha = check_tensor(alpha)
    gamma = check_tensor(gamma)

    n_drivers = gamma.shape[0]
    kernels = []
    for i in range(n_drivers):
        kernel = gamma * torch.exp(- gamma * t)
        kernel = kernel + 0
        mask_kernel = (t < lower) | (t > upper)
        kernel[mask_kernel] = 0.
        kernel /= (kernel.sum() * dt)
        kernel *= alpha[i]
        kernels.append(kernel)

    return torch.stack(kernels, 0).float()


def compute_kernels(t, alpha, m, sigma=None, kernel_name='raised_cosine',
                    lower=None, upper=None, dt=None):
    """

    Returns
    -------
    kernels : torch.Tensor

    """
    if kernel_name == 'gaussian':
        assert sigma is not None
        kernels = truncated_gaussian_kernel(
            t, alpha, m, sigma, lower, upper, dt)
    elif kernel_name == 'raised_cosine':
        assert sigma is not None
        kernels = raised_cosine_kernel(
            t, alpha, m, sigma)
    elif kernel_name == 'exponential':
        kernels = exponential_kernel(
            t, alpha, m, lower, upper, dt)
    else:
        raise ValueError(
            f"kernel_name must be 'gaussian' | 'raised_cosine',"
            " got '{kernel_name}'"
        )
    return kernels
