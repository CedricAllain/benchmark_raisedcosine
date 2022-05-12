import numpy as np
import numbers
import torch


def check_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    return x


def check_array_size(var, n=1, repeat=True):
    """Take a variable and make it an array of length n.

    Parameters
    ----------
    var : int | float | array-like
        The variable we want to convert.

    n : int
        The length of the return array. Defauls to 1.

    repeat : bool
        If True, if var is of dimension 1 and n > 1, return an array of
        repeated var of length n. Defaults to True.

    Returns
    -------
    1d-array
    """
    var = np.atleast_1d(var)

    if len(var) == n:  # already at the good size
        return var
    elif repeat and (n > 1) and (len(var) == 1):
        var = np.repeat(var, n)

    assert len(var) == n, \
        "var must be an int, float or an array of length %i, but got %s" % (
            n, var)

    return var


def check_driver_tt(driver_tt):
    """Ensure that driver_tt is a list of 1d-arrays"""

    if driver_tt is None:
        return [np.array([])]

    if isinstance(driver_tt[0], numbers.Number):
        driver_tt = [driver_tt]

    driver_tt = [np.array(x) for x in driver_tt]

    return driver_tt


def check_acti_tt(acti_tt):
    """Ensure that acti_tt is a 1d-array"""
    if acti_tt is None:
        return np.array([])

    return np.atleast_1d(acti_tt)


def kernel_intensity(baseline, driver, kernels, L=100):
    """
    Parameters
    ----------
    baseline : float
        baseline intensity parameter

    driver : sparse vector where non nul values correspond to driver timestamps

    kernel : vector of length L, the kernel values over its support

    L : int
        kernel times

    Returns
    -------
    intensity : torch.Tensor
    """
    intensity = baseline + torch.conv_transpose1d(
        driver[None], kernels[:, None]
    )[0, 0, :-L+1]
    # intensity = torch.concat([torch.zeros(1), intensity])
    return intensity.clip(0)
