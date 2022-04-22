##########################################
# Simulation of driver and activation functions
# from Inhomogeneous Poisson process
##########################################

import numpy as np

from tick.base import TimeFunction
from tick.hawkes import SimuHawkes
from tick.hawkes import HawkesKernelTimeFunc
from tick.hawkes import SimuInhomogeneousPoisson
from tick.plot import plot_point_process

from kernels import raised_cosine_kernel


def simu(true_params, simu_params=[50, 1000, 0.5], isi=0.7, seed=None,
         plot_intensity=True, reparam=False):
    """Simulate drivers and intensity timestamps

    Parameters
    ----------
    true_params = 2darray-like
        ex.: params = nn.tensor([[1, np.nan],  # baseline
                                 [0.7, 0.5],   # alpha
                                 [0.4, 0.6],   # m
                                 [0.4, 0.2]])  # sigma

    Returns
    -------
    intensity_csc

    z : array-like
        sparse vector where 1 indicates an intensity activation
    """

    intensity_value = true_params[0][0]
    n_drivers = true_params.shape[1]

    # simulate driver events
    kernel_value = []
    driver_tt = []
    driver = []

    # XXX: here only between 0 and 1
    t_value = np.linspace(0, 1, L + 1)[:-1]

    for i in range(n_drivers):
        T, L, p_task = simu_params
        dt = 1 / L

        this_kernel_value = np.array(raised_cosine_kernel(
            t_value, true_params[1:, i], reparam))
        kernel_value.append(this_kernel_value)

        # generate driver timestamps
        grid_tt = np.arange(start=0, stop=T - 2 * isi, step=isi)
        # sample timestamps
        rng = np.random.RandomState(seed=seed)
        this_driver_tt = rng.choice(grid_tt, size=int(p_task * len(grid_tt)),
                                    replace=False).astype(float)
        this_driver_tt = (this_driver_tt / dt).astype(int) * dt
        this_driver_tt.sort()
        driver_tt.append(this_driver_tt)
        # create sparse vector
        t = np.arange(0, T + 1e-10, dt)
        this_driver = t * 0
        this_driver[(this_driver_tt * L).astype(int)] += 1
        driver.append(this_driver)
        intensity_value += np.convolve(this_driver, this_kernel_value)[:-L+1]

    kernel_value = np.array(kernel_value)
    driver_tt = np.array(driver_tt)
    driver = np.array(driver)

    # Simulate intensity events
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

    acti_tt = (in_poi.timestamps[0] / dt).astype(int) * dt
    acti_tt.sort()
    acti = t * 0
    acti[(acti_tt * L).astype(int)] += 1

    return kernel_value, intensity_value, driver_tt, driver, acti_tt, acti
