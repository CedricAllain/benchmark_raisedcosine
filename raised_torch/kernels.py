##########################################
# Define kernel functions
##########################################

import numpy as np
import torch
from utils_plot import check_tensor


def raised_cosine_kernel(t, params, dt=1/1000, kernel_zero_base=False):
    """Compute the raised cosine distribution kernel. 
    """

    t = check_tensor(t)
    params = check_tensor(params)

    _, alpha, mu, sig = params

    kernel = (1 + torch.cos((t - mu) / sig * np.pi)) / (2 * sig)
    mask_kernel = (t < (mu - sig)) | (t > (mu + sig))
    kernel[mask_kernel] = 0.
    if kernel_zero_base:
        kernel = (kernel - kernel.min())
    kernel = alpha * kernel 

    return kernel


def truncated_gaussian_kernel(t, params, lower, upper, 
                            dt=1/1000, kernel_zero_base=False):
    """Compute the truncated normal distribution kernel. 
    """

    t = check_tensor(t)
    params = check_tensor(params)

    _, alpha, mu, sig = params

    kernel = torch.exp(-(t - mu) ** 2 / sig ** 2)
    mask_kernel = (t < lower) | (t > upper)
    kernel[mask_kernel] = 0.
    if kernel_zero_base:
        kernel = (kernel - kernel.min())
    kernel = alpha * kernel / (kernel.sum() * dt)

    return kernel

