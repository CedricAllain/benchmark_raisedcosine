# %%
from scipy.stats import truncnorm
import numpy as np
from numpy.testing import assert_almost_equal
import torch
import pytest
import matplotlib.pyplot as plt

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
def test_truncated_gaussian_kernel(n, plot=False):
    dt = 1/(10**n)
    t = torch.arange(0, 1, dt)

    alpha = np.array([1, 0.8])
    m = np.array([0.4, 0.9])
    sigma = np.array([0.2, 0.4])
    lower, upper = 0, 1

    kernels = np.array(truncated_gaussian_kernel(
        t, alpha, m, sigma, lower, upper))

    # assert the integral is one
    assert_almost_equal(kernels.sum(axis=1)*dt, alpha, n)

    # max
    a = (lower - m) / sigma
    b = (upper - m) / sigma
    true_max = truncnorm.pdf(m, a, b, loc=m, scale=sigma) * alpha
    assert_almost_equal(kernels.max(axis=1), true_max, n)

    # compare to exact values
    true_kernel = []
    for i in range(2):
        this_true_kernel = truncnorm.pdf(
            np.arange(0, 1, dt), a[i], b[i], loc=m[i], scale=sigma[i])
        this_true_kernel *= alpha[i]
        true_kernel.append(this_true_kernel)
        if plot:
            plt.plot(t, kernels[i], label='torch')
            plt.plot(t, this_true_kernel, label='dripp')
            plt.legend()
            plt.title(f'kernel {i+1}')
            plt.show()

    true_kernel = np.array(true_kernel)

    assert_almost_equal(kernels, true_kernel, n)


for n in [2, 3, 4]:
    print(n)
    test_truncated_gaussian_kernel(n, plot=True)

# %%
