##########################################
# Define the class of intensity function
##########################################

import numpy as np
from tqdm import tqdm
import time
import torch
from torch import nn


from kernels import raised_cosine_kernel, truncated_gaussian_kernel




class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.

    Parameters
    ----------
    t : XXX

    params : tuple
        model parameters (baseline, alpha, mu, sigma)

    dt : XXX

    kernel_name : str, 'gaussian' | 'raised_cosine'
        name of 

    loss_name : XXX

    """

    def __init__(self, t, params, dt,
                 kernel_name, loss_name):

        super().__init__()

        self.weights = nn.Parameter(params)
        self.t = t
        self.dt = dt
        self.L = len(self.t)
        self.kernel_name = kernel_name
        self.loss_name = loss_name

    def forward(self, driver_tt_torch):
        """Function to be optimised (the intensity).,

        Parameters
        ----------
        driver_tt_torch : XXX

        Returns
        -------
        intensity : XXX
        """

        mu_0, alpha, mu, sig = self.weights

        if self.kernel_name == 'gaussian':
            self.kernel = truncated_gaussian_kernel(self.t, self.weights)
        elif self.kernel_name == 'raised_cosine':
            self.kernel = raised_cosine_kernel(self.t, self.weights)
        else:
            raise ValueError(
                f"kernel_name must be 'gaussian' | 'raised_cosine',"
                " got '{self.kernel_name}'"
            )

        intensity = mu_0 + torch.conv_transpose1d(driver_tt_torch[None, None],
                                                  self.kernel[None, None],
                                                  )[0, 0, :-self.L+1]

        return intensity
