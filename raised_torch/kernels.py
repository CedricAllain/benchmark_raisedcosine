##########################################
# Define kernel functions
##########################################

import numpy as np
import torch

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


def truncated_gaussian_kernel(t, alpha, m, sigma, lower, upper):
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
    dt = t[1] - t[0]
    alpha = check_tensor(alpha)
    m = check_tensor(m)
    sigma = check_tensor(sigma)

    n_drivers = m.shape[0]
    kernels = []
    for i in range(n_drivers):
        kernel = torch.exp((- torch.square(t - m[i]) /
                            (2 * torch.square(sigma[i]))))
        kernel = kernel + 0
        mask_kernel = (t < lower) | (t > upper)
        kernel[mask_kernel] = 0.
        kernel /= (kernel.sum() * dt)
        kernel *= alpha[i]
        kernels.append(kernel)

    return torch.stack(kernels, 0).float()


def compute_kernels(t, alpha, m, sigma, kernel_name='raised_cosine',
                    lower=None, upper=None):
    """

    Returns
    -------
    kernels : torch.Tensor

    """
    if kernel_name == 'gaussian':
        kernels = truncated_gaussian_kernel(
            t, alpha, m, sigma, lower, upper)
    elif kernel_name == 'raised_cosine':
        kernels = raised_cosine_kernel(
            t, alpha, m, sigma)
    else:
        raise ValueError(
            f"kernel_name must be 'gaussian' | 'raised_cosine',"
            " got '{kernel_name}'"
        )
    return kernels
