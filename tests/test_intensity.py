# %%
import numpy as np
import torch
from numpy.testing import assert_almost_equal
import pytest

from raised_torch.kernels import *
from raised_torch.utils.utils import kernel_intensity
from raised_torch.simu_pp import simu_driver_tt


@pytest.mark.parametrize("L", [100, 200, 500, 1_000])
def test_kernel_intensity(L):
    T = 1_000
    dt = 1/L
    t = torch.arange(0, 1, dt)

    m = np.array([0.4, 0.8])
    sigma = np.array([0.2, 0.05])
    lower, upper = 0, 0.8
    isi = [1, 1.4]

    def procedure(baseline, alpha):
        kernel_name = 'gaussian'
        kernels = compute_kernels(
            t, alpha, m, sigma, kernel_name, lower, upper)
        # simulate driver timestamps
        driver, driver_tt = simu_driver_tt(
            isi, len(alpha), p_task=0.6, T=T, L=L, seed=0)
        intensity = kernel_intensity(
            baseline, check_tensor(driver), kernels, L)
        # compute integral estimation
        integ_est = intensity.sum() * dt
        # close form of the integral
        integ = baseline * T
        for this_alpha, this_driver_tt in zip(alpha, driver_tt):
            integ += this_alpha * len(this_driver_tt)

        assert_almost_equal(integ_est, integ, 2)

    procedure(baseline=1, alpha=np.array([0, 0]))
    procedure(baseline=0, alpha=np.array([1., 1.]))
    procedure(baseline=1, alpha=np.array([1., 1.]))

# %%
