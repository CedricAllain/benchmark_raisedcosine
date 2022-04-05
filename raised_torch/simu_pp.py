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



def simu(true_params, simu_params=[50, 1000, 0.5], seed=None,
         plot_intensity=True,):
    """

    Parameters
    ----------

    Returns
    -------
    intensity_csc

    z : array-like
        sparse vector where 1 indicates an intensity activation
    """

    mu_0, alpha_true, mu_true, sig_true = true_params
    T, L, p_task = simu_params
    dt = 1 / L

    # simulate data
    t_value = np.linspace(0, 1, L + 1)[:-1]
    y_value = np.array(raised_cosine_kernel(t_value, true_params, dt))

    # generate driver timestamps
    isi = 0.7
    t_k = np.arange(start=0, stop=T - 2 * isi, step=isi)
    # sample timestamps
    rng = np.random.RandomState(seed=seed)
    t_k = rng.choice(t_k, size=int(p_task * len(t_k)),
                     replace=False).astype(float)
    t_k = (t_k / dt).astype(int) * dt
    # create sparse vector
    t = np.arange(0, T + 1e-10, dt)
    driver_tt = t * 0
    driver_tt[(t_k * L).astype(int)] += 1
    intensity_csc = mu_0 + np.convolve(driver_tt, y_value)[:-L+1]

    #
    tf = TimeFunction((t, intensity_csc), dt=dt)
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

    t_k = (in_poi.timestamps[0] / dt).astype(int) * dt
    acti_tt = t * 0
    acti_tt[(t_k * L).astype(int)] += 1

    return y_value, intensity_csc, driver_tt, acti_tt, in_poi