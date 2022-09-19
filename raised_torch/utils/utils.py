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


def grid_projection(events, L, remove_duplicates=False, verbose=False):
    """Project on a grid of step 1/L the given events

    Parameters
    ----------
    events : array-like
        the event timestamps to project on the grid

    L : int
        grid frequency, e.g., with L = 100, the grid will be defined with a
        step of 0.01

    remove_duplicates : bool
        if True, remove duplicates renders by grid projection

    verbose : bool

    Returns
    -------
    events_grid : numpy array
        the input events projected on the given grid
    """
    def procedure(events):
        events_grid = np.array([np.round(tt * L).astype(int) / L
                                for tt in events])
        if remove_duplicates:
            events_grid = np.unique(events_grid)
            diff_len = len(events) - len(events_grid)
            p_lost = diff_len / len(events)
            if verbose:
                print(f"grid projection (L = {L}) renders {diff_len} duplicates "
                      f"({p_lost * 100}%) that were removed.")

        return events_grid

    if type(events[0]) is not np.ndarray:
        return procedure(events)
    else:  # events is a multidim array
        return [procedure(this_events) for this_events in events]


def get_sparse_from_tt(events, T, dt):
    """
    From a array of timestamps, get the associated sparse vector
    """

    def procedure(events):
        sparse = np.arange(0, T + 1e-10, dt) * 0
        sparse[(events / dt).astype(int)] += 1
        return sparse

    if type(events[0]) is not np.ndarray:
        return procedure(events)
    else:  # events is a multidim array
        return [procedure(this_events) for this_events in events]


def get_rc_std(sigma):
    """
    variance = s^2 (1/3 - 2/pi^2)
    where s defines the raised cosine distribution support: [mu-s, mu+s]
    """
    return np.sqrt(sigma**2 * (1/3 - 2/np.pi**2))


def get_non_param_estimation(kernel_support, kernel_size, acti_tt, driver_tt):
    """
    Parameters:
    -----------

    kernel_size : int
    """

    from tick.hawkes import HawkesEM

    em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size,
                  n_threads=8, verbose=False, tol=1e-3, max_iter=100)
    events = [acti_tt]
    for this_driver_tt in driver_tt:
        events.append(this_driver_tt)
        
    em.fit(events)

    return em
