##########################################
# Define kernel functions
##########################################

import numpy as np
import torch
from utils_plot import check_tensor


def raised_cosine_kernel(t, params, reparam=False):
    """Compute the raised cosine distribution kernel. 

    Parameters
    ----------
    t : tensor | array-like
        timepoints to compute kernel value at

    params : tensor | tuple
        model parameters (baseline, alpha, mu, sigma)

    reparam : bool
        if True, apply u = m - sigma reparametrization

    Returns
    -------
    tensor
    """

    t = check_tensor(t)
    params = check_tensor(params)

    if reparam:
        alpha, u, sig = params[-3:]  # u = mu - sig
        kernel = (1 + torch.cos((t - u) / sig * np.pi - np.pi)) / (2 * sig)
        mask_kernel = (t < u) | (t > (u + 2*sig))
    else:
        alpha, m, sig = params[-3:]
        kernel = (1 + torch.cos((t - m) / sig * np.pi)) / (2 * sig)
    mask_kernel = (t < (m - sig)) | (t > (m + sig))

    kernel[mask_kernel] = 0.
    kernel = alpha * kernel

    return kernel


def truncated_gaussian_kernel(t, params, lower, upper):
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
    params = check_tensor(params)

    _, alpha, mu, sig = params

    kernel = torch.exp(-(t - mu) ** 2 / sig ** 2)
    mask_kernel = (t < lower) | (t > upper)
    kernel[mask_kernel] = 0.
    kernel = alpha * kernel

    return kernel
