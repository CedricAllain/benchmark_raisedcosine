import torch
import pytest


@pytest.mark.parametrize("n_drivers", [10, 20])
def test_conv1d(n_drivers):
    x = torch.randn(n_drivers, 100)  # n_drivers x n_times
    k = torch.randn(n_drivers, 20)   # n_drivers x n_times_driver

    res = torch.conv_transpose1d(x[None], k[:, None])

    assert res.shape == (1, 1, 119)
