"""
Plot, using tick methods, the computation time as a function of the problem
size, to show that with non parametrized kernel, tick is slow with
discretization.
"""
# %% Packages
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time

from joblib import Memory, Parallel, delayed

from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp,
                         HawkesEM)
from tick.base import TimeFunction
from tick.plot import plot_hawkes_kernels, plot_point_process

from raised_torch.model import Model
from raised_torch.kernels import compute_kernels
from raised_torch.solver import initialize, training_loop
from raised_torch.utils.utils import check_tensor, kernel_intensity

N_JOBS = 4
# %%
T = 100
L = 100
max_iter = 1_000
N_JOBS = 4
dt = 1/L
lower, upper = 0, 1
t_values = np.arange(lower, upper, dt)
baseline = 1
alpha = 0.5
gamma = 5
EPS = 0.01
assert np.exp(-gamma * upper) < EPS
exp_kernel = alpha * gamma * np.exp(-gamma * t_values)

# %% Plot exponential kernel
plt.plot(t_values, exp_kernel, label="True kernel")
plt.hlines(EPS, lower, upper, color='k', linestyles='--', label="EPS")
plt.legend()
plt.title(f"True exponential kernel with gamma={gamma}")
plt.xlabel('t')
plt.xlim(lower, upper)
plt.ylim(0, None)
plt.show()


def procedure(T, L, seed=0, plot=False, verbose=False):
    # Simulate data
    hawkes = SimuHawkes(baseline=np.array([baseline]), end_time=T, verbose=False,
                        seed=None, force_simulation=True)
    hawkes.set_kernel(0, 0, HawkesKernelExp(alpha, gamma))
    hawkes.track_intensity(dt)
    start = time.time()
    if verbose:
        print(f"Simulate data with Tick...\r", end='', flush=True)
    hawkes.simulate()
    end_time = time.time() - start
    if verbose:
        print(f"Simulate data with Tick... done ({end_time:.3f} s.) ")

    intensity = hawkes.tracked_intensity
    # intensity_times = hawkes.intensity_tracked_times
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        fig = plot_point_process(
            hawkes, n_points=50000, t_min=0, t_max=5, ax=ax)
        ax.set_ylim([0, None])
        ax.set_xlim([0, None])
        plt.show()

    driver_tt = hawkes.timestamps

    # Learn with tick
    em = HawkesEM(upper, kernel_size=(upper*L), n_threads=N_JOBS, verbose=True,
                  tol=1e-3, max_iter=max_iter, record_every=1)

    start = time.time()
    if verbose:
        print(f"Fitting model...\r", end='', flush=True)
    em.fit(driver_tt)
    tick_time = time.time() - start
    if verbose:
        print(f"Fitting model... done ({tick_time:.3f} s.) ")

    tick_history = em.get_history()
    if verbose:
        print(f'Tick solved in {len(tick_history['n_iter'])} iterations')
        print(f'Final log-likelihood with Tick: {em.score(driver_tt)}')

    if plot:
        fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)
        plt.ylim([0, None])
        plt.xlim([lower, upper])
        plt.show()
    # Learn with torch

    driver = intensity[0] * 0
    driver[np.round(driver_tt[0] * L).astype(int)] += 1
    driver = check_tensor(np.array([driver]))

    # kernels = compute_kernels(t_values, [alpha], m=[gamma],
    #                         kernel_name='exponential', lower=lower, upper=upper)
    # intensity_torch = kernel_intensity(baseline, driver, kernels, L)
    # plt.plot(t_values, kernels[0])
    # plt.xlabel('t')
    # plt.xlim(lower, upper)
    # plt.ylim(0, None)
    # plt.show()

    # initialize parameters close to true values
    baseline_init = intensity[0][0] + np.random.normal(0, 0.5)
    alpha_init = [max(alpha + np.random.normal(0, 0.5), EPS)]
    gamma_init = [gamma + np.random.normal(0, 0.5)]
    sigma_init = None
    # init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
    #                          lower=lower, upper=upper,
    #                          kernel_name=kernel_name)
    # baseline_init, alpha_init, m_init, sigma_init = init_params
    if verbose:
        print('initial parameters (baseline, alpha, gamma)',
              baseline_init, alpha_init, gamma_init)

    t = torch.arange(lower, upper, dt)
    model_raised = Model(t, baseline_init, alpha_init, gamma_init,
                         sigma_init, dt=dt,
                         kernel_name="exponential", loss_name='log-likelihood',
                         lower=lower, upper=upper)
    if verbose:
        print(
            f'Discretization makes lose {int(len(driver_tt[0])-driver.sum())} driver events')

    res_dict = training_loop(model_raised, driver, driver[0], solver="RMSprop",
                             step_size=1e-3, max_iter=max_iter, test=False,
                             logging=plot, device='cpu')
    torch_time = res_dict['compute_time']

    if plot:
        df_hist = pd.DataFrame(res_dict['hist'])
        plt.plot(df_hist['time_loop'], df_hist['loss'], label=method)
        plt.show()

    return dict(T=T, L=L, tick=tick_time, torch=torch_time)


procedure(T=1_000, L=100, seed=0, plot=True, verbose=True)
# %%
T_list = [100, 500, 1_000, 5_000, 10_000]
df_T = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(this_T, L, seed=0) for this_T in T_list)
df_T = pd.DataFrame(df_T)
print(df_T)
# %%

for method in ['tick', 'torch']:
    plt.plot(df_T['T'], df_T[method], label=method)
plt.xlim(min(T_list), max(T_list))
plt.xlabel('T')
plt.ylabel('Computation time (s.)')
plt.xscale('log')
plt.xticks(T_list)
plt.title('Computation time')
plt.legend()
plt.show()
# %%
L_list = [100, 200, 500, 1_000]
df_L = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(T, this_L) for this_L in L_list)
df_L = pd.DataFrame(df_L)

for method in ['tick', 'torch']:
    plt.plot(df_L['L'], df_L[method], label=method)
plt.xlim(min(L_list), max(L_list))
plt.xlabel('L')
plt.ylabel('Computation time (s.)')
plt.xticks(L_list)
plt.title('Computation time')
plt.legend()
plt.show()
# %%
