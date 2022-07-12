##########################################
# Define the class of intensity function
##########################################
import numpy as np
import torch
from torch import nn


from .kernels import compute_kernels
from .utils.utils import check_tensor, kernel_intensity


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

    def __init__(self, t, baseline, alpha, m, sigma, dt=1/100,
                 kernel_name='raised_cosine', loss_name='log-likelihood',
                 lower=None, upper=None, driver=None):

        super().__init__()

        self.kernel_name = kernel_name

        self.baseline = nn.Parameter(check_tensor(baseline))
        self.alpha = nn.Parameter(check_tensor(alpha))
        if (self.kernel_name == 'gaussian') or (self.kernel_name == 'exponential'):
            self.m = nn.Parameter(check_tensor(m))
        elif self.kernel_name == 'raised_cosine':
            # reparametrazion, u = m - sigma
            self.m = nn.Parameter(check_tensor(m) - check_tensor(sigma))
        else:
            raise ValueError(
                "kernel_name must be 'gaussian' | 'raised_cosine' | 'exponential',"
                f" got '{self.kernel_name}'"
            )

        if loss_name in ['MLE', 'log-likelihood']:
            self.loss_name = loss_name
        else:
            raise ValueError(
                f"loss must be 'MLE' or 'log-likelihood', got '{loss_name}'"
            )

        if sigma is not None:
            self.sigma = nn.Parameter(check_tensor(sigma))
        else:
            self.sigma = None

        self.lower = lower
        self.upper = upper
        self.register_buffer('t', t)
        self.dt = dt
        self.L = len(self.t)

        self.n_driver_events = None
        if driver is not None:
            self.n_driver_events = [this_driver.sum().item()
                                    for this_driver in driver]

        # compute initial kernels
        self.kernels = compute_kernels(
            self.t, self.alpha, self.m, self.sigma, self.kernel_name,
            self.lower, self.upper, self.dt)

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

        # compute updated kernels
        self.kernels = compute_kernels(
            self.t, self.alpha, self.m, self.sigma, self.kernel_name,
            self.lower, self.upper, self.dt)

        return kernel_intensity(self.baseline, driver, self.kernels, self.L)

    def compute_loss(self, intensity, acti, T, driver=None):
        """

        Parameters
        ----------
        acti : torch.Tensor
            sparse tensor, 1 where there is an activation

        Returns
        -------
        XXX
        """
        if self.loss_name == 'log-likelihood':
            # first evaluate intensity integral
            nll = self.baseline * T
            n_driver_events = self.n_driver_events
            if driver is not None:
                n_driver_events = [this_driver.sum().item()
                                   for this_driver in driver]
            for this_alpha, this_n in zip(self.alpha, n_driver_events):
                nll += this_alpha * this_n
            # negative log-likelihood
            return (nll - torch.log(intensity[acti]).sum())/T
        elif self.loss_name == 'MSE':
            return ((intensity ** 2).sum() * self.dt
                    - 2 * (intensity[acti]).sum())/T
