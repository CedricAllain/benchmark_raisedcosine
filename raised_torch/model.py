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

    def __init__(self, t, params, reparam=False, dt=1/100,
                 kernel_name='raised_cosine', loss_name='log-likelihood'):

        super().__init__()

        self.kernel_name = kernel_name
        self.reparam = reparam

        if self.kernel_name == 'raised_cosine' and self.reparam:
            # reparametrization, u = (mu - sigma)
            params[2] = params[2] - params[3]

        self.weights = nn.Parameter(params)
        # self.alpha = nn.Parameter(alpha)  # alpha 1darray
        self.n_drivers = self.weights.shape[1]


        self.t = t
        self.dt = dt
        self.L = len(self.t)

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

        intensity = self.weights[0][0]  # baseline

        self.kernel = []
        for i in range(self.n_drivers):
            if self.kernel_name == 'gaussian':
                this_kernel = truncated_gaussian_kernel(
                    self.t, self.weights[1:, i])
            elif self.kernel_name == 'raised_cosine':
                this_kernel = raised_cosine_kernel(
                    self.t, self.weights[1:, i], self.reparam)
            else:
                raise ValueError(
                    f"kernel_name must be 'gaussian' | 'raised_cosine',"
                    " got '{self.kernel_name}'"
                )
            self.kernel.append(this_kernel)
            intensity += torch.conv_transpose1d(driver_tt_torch[None, None],
                                                this_kernel[None, None],
                                                )[0, 0, :-self.L+1]

        return intensity
