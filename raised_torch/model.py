##########################################
# Define the class of intensity function
##########################################

import numpy as np
from tqdm import tqdm
import time
import torch
from torch import nn


from .kernels import raised_cosine_kernel, truncated_gaussian_kernel
from .utils.utils import check_tensor


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.

    Parameters
    ----------
    t : XXX

    params : 2darray-like
        ex.: params = nn.tensor([[1, np.nan],  # baseline
                                 [0.7, 0.5],   # alpha
                                 [0.4, 0.6],   # m
                                 [0.4, 0.2]])  # sigma

    dt : XXX

    kernel_name : str, 'gaussian' | 'raised_cosine'
        name of 

    loss_name : XXX

    """

    def __init__(self, t, baseline, alpha, m, sigma, reparam=False, dt=1/100,
                 kernel_name='raised_cosine', loss_name='log-likelihood'):

        super().__init__()

        self.kernel_name = kernel_name
        self.reparam = reparam

        self.baseline = nn.Parameter(check_tensor(baseline))
        self.alpha = nn.Parameter(check_tensor(alpha))
        
        self.sigma = nn.Parameter(check_tensor(sigma))
        if self.kernel_name == 'gaussian':
            self.m = nn.Parameter(check_tensor(m))
        elif self.kernel_name == 'raised_cosine':
            self.m = nn.Parameter(check_tensor(m), requires_grad=False)
            # reparametrazion for raised cosine, u = m - sigma
            self.u = nn.Parameter(self.m - self.sigma)
        else:
            raise ValueError(
                f"kernel_name must be 'gaussian' | 'raised_cosine',"
                " got '{self.kernel_name}'"
            )

        self.t = t

        # compute initial kernels
        self.compute_kernels()
        
        self.dt = dt
        self.L = len(self.t)
        self.loss_name = loss_name
        
    def compute_kernels(self):
        if self.kernel_name == 'gaussian':
            self.kernels = truncated_gaussian_kernel(
                self.t, self.alpha, self.m, self.sigma)
        elif self.kernel_name == 'raised_cosine':
            self.kernels = raised_cosine_kernel(
                self.t, self.alpha, self.u, self.sigma)
        else:
            raise ValueError(
                f"kernel_name must be 'gaussian' | 'raised_cosine',"
                " got '{self.kernel_name}'"
            )

    def forward(self, driver):
        """Function to be optimised (the intensity).,

        Parameters
        ----------
        driver : 2d torch.Tensor
            size n_drivers, n_times

        Returns
        -------
        intensity : XXX
        """

        self.compute_kernels()

        intensity = self.baseline + torch.conv_transpose1d(
            check_tensor(driver)[None],
            self.kernels[:, None])[0, 0, :-self.L+1]

        return intensity
