# %%
from scipy.stats import truncnorm
import numpy as np
from numpy.testing import assert_almost_equal
import torch
import pytest

from raised_torch.kernels import *


def test_raised_cosine_kernel():
    dt = 1/100
    t = torch.arange(0, 1, dt)

    alpha = np.array([1, 0.8])
    m = np.array([0.4, 0.6])
    sigma = np.array([0.2, 0.4])
    u = m - sigma
    kernels = np.array(raised_cosine_kernel(
        t, alpha, u, sigma))

    assert_almost_equal(kernels.sum(axis=1)*dt, alpha)
    assert_almost_equal(kernels.max(axis=1), 1/sigma * alpha)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_truncated_gaussian_kernel(n):
    dt = 1/(10**n)
    t = torch.arange(0, 1, dt)

    alpha = np.array([1, 0.8])
    m = np.array([0.4, 0.6])
    sigma = np.array([0.2, 0.4])
    lower, upper = 0, 0.8

    kernels = np.array(truncated_gaussian_kernel(
        t, alpha, m, sigma, lower, upper))

    assert_almost_equal(kernels.sum(axis=1)*dt, alpha)
    # max
    a = (lower - m) / sigma
    b = (upper - m) / sigma
    true_max = truncnorm.pdf(m, a, b, loc=m, scale=sigma) * alpha
    assert_almost_equal(kernels.max(axis=1), true_max, n)

# %%
