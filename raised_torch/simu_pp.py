##########################################
# Simulation of driver and activation functions
# from Inhomogeneous Poisson process
##########################################

import numpy as np
import torch

from tick.base import TimeFunction
from tick.hawkes import SimuHawkes, SimuPoissonProcess
from tick.hawkes import HawkesKernelTimeFunc
from tick.hawkes import SimuInhomogeneousPoisson
from tick.plot import plot_point_process

from .kernels import compute_kernels
from .utils.utils import check_tensor, check_array_size, check_driver_tt, \
    check_acti_tt, kernel_intensity


def simu_driver_tt(isi, n_drivers, p_task, T, L, seed=0):
    isi = check_array_size(isi, n=n_drivers)
    dt = 1 / L
    # simulate driver events
    driver_tt = []
    driver = []
    for i in range(n_drivers):  # n_drivers = m.shape[0]
        # generate driver timestamps sampling grid
        grid_tt = np.arange(start=0, stop=(T-2*isi[i]), step=isi[i])
        # sample timestamps
        rng = np.random.RandomState(seed=seed+i)
        this_driver_tt = rng.choice(grid_tt, size=int(p_task * len(grid_tt)),
                                    replace=False).astype(float)
        this_driver_tt = np.unique(
            np.round(this_driver_tt / dt).astype(int) * dt)
        this_driver_tt.sort()
        driver_tt.append(this_driver_tt)
        # create sparse vector
        t = np.arange(0, T + 1e-10, dt)
        this_driver = t * 0
        this_driver[np.round(this_driver_tt * L).astype(int)] += 1
        driver.append(this_driver)

    return np.array(driver), driver_tt


def simu(baseline, alpha, m, sigma, kernel_name='raised_cosine',
         simu_params=[50, 1000, 0.5], isi=0.7, seed=42,
         plot_intensity=False, lower=None, upper=None):
    """Simulate drivers and intensity timestamps

    Parameters
    ----------
    true_params = 2darray-like
        ex.: params = nn.tensor([[1, np.nan],  # baseline
                                 [0.7, 0.5],   # alpha
                                 [0.4, 0.6],   # m
                                 [0.4, 0.2]])  # sigma

    simu_params : 1d array
        T, L, p_task

    Returns
    -------
    intensity_csc

    z : array-like
        sparse vector where 1 indicates an intensity activation
    """

    T, L, p_task = simu_params
    dt = 1 / L

    baseline = check_tensor(baseline)
    alpha = check_tensor(alpha)
    m = check_tensor(m)
    sigma = check_tensor(sigma)
    if kernel_name == 'raised_cosine':
        m -= sigma

    # XXX: here only between 0 and 1
    t_value = np.linspace(0, 1, L + 1)[:-1]
    kernels = compute_kernels(
        t_value, alpha, m, sigma, kernel_name, lower, upper, dt)

    driver, driver_tt = simu_driver_tt(
        isi, n_drivers=len(alpha), p_task=p_task, T=T, L=L, seed=seed)

    driver = check_tensor(driver)
    intensity_value = kernel_intensity(baseline, driver, kernels, L)

    # Simulate intensity events
    t = np.arange(0, T + 1e-10, dt)
    tf = TimeFunction((t, intensity_value), dt=dt)
    # We define a 1 dimensional inhomogeneous Poisson process with the
    # intensity function seen above
    in_poi = SimuInhomogeneousPoisson(
        [tf], end_time=T, seed=seed, verbose=False)
    # We activate intensity tracking and launch simulation
    in_poi.track_intensity(dt)
    in_poi.simulate()

    # We plot the resulting inhomogeneous Poisson process with its
    # intensity and its ticks over time
    if plot_intensity:
        plot_point_process(in_poi)

    acti_tt = np.unique(np.round(in_poi.timestamps[0] / dt).astype(int) * dt)
    acti_tt.sort()
    acti = t * 0
    acti[np.round(acti_tt * L).astype(int)] += 1

    return kernels, intensity_value, check_driver_tt(driver_tt), driver, \
        check_acti_tt(acti_tt), acti
