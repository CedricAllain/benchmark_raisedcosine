"""
Plot, as a function of the grid discretization step, the convergence of final
estimates compared to the estimates using EM in a continuous setting.
"""

# %%
import numpy as np
import pandas as pd
from sqlalchemy import true
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tick.hawkes import HawkesEM

from raised_torch.simu_pp import simu
from raised_torch.model import Model
from raised_torch.solver import initialize, training_loop, compute_loss, optimizer
from raised_torch.kernels import compute_kernels
from raised_torch.utils.utils import grid_projection, check_tensor, get_sparse_from_tt
from raised_torch.utils.utils_plot import plot_hist_params

from trunc_norm_kernel.simu import simulate_data
from trunc_norm_kernel.optim import em_truncated_norm
from trunc_norm_kernel.model import TruncNormKernel, Intensity
from trunc_norm_kernel.metric import negative_log_likelihood

# %% true parameters
n_drivers = 2
baseline = 1
alpha = [1., 1.]
m = [0.4, 0.8]
sigma = [0.2, 0.05]
lower, upper = 0, 1
true_params = {'baseline': baseline, 'alpha': alpha, 'm': m, 'sigma': sigma}

# %% simulation parameters
T = 10_000
p_task = 0.4
isi = [1.2, 1.2]
seed = 42
driver_tt, acti_tt, kernel, intensity = simulate_data(
    lower=lower, upper=upper, m=m, sigma=sigma, sfreq=None, baseline=baseline,
    alpha=alpha, T=T, isi=isi, add_jitter=True, n_tasks=p_task,
    n_drivers=n_drivers, seed=seed, return_nll=False, verbose=True)

# %% initialize parameters
kernel_name = 'gaussian'
init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
                         lower=lower, upper=upper,
                         kernel_name=kernel_name)
baseline_init, alpha_init, m_init, sigma_init = init_params
init_params[0] = [baseline_init, np.nan]

# %% varying discretization step, learn params with torch model
loss_name = 'log-likelihood'
max_iter = 200
solver = 'Adam'
step_size = 1e-3
L_list = [10, 20, 25, 40, 50, 100, 200, 250, 400,
          500, 1000, 2000, 2500, 4000, 5000, 10000]
# L_list = [100, 200]
dict_hist = {}
for this_L in L_list:
    print(f"this_L: {this_L}")
    # grid projection of timestamps
    driver_tt_ = grid_projection(driver_tt, this_L, remove_duplicates=False)
    acti_tt_ = grid_projection(acti_tt, this_L, remove_duplicates=False)
    # make sparse vector from timestamps
    dt = 1 / this_L
    driver = get_sparse_from_tt(driver_tt_, T, dt)
    acti = get_sparse_from_tt(acti_tt_, T, dt)
    # Learn with torch solver
    t = torch.arange(lower, upper, dt)  # maximal kernel support
    model = Model(t, baseline_init, alpha_init, m_init, sigma_init,
                  dt, kernel_name, loss_name, lower, upper, driver)
    res_dict = training_loop(model, driver, acti, T=T, solver='RMSProp',
                             step_size=1e-3, max_iter=100, test=False,
                             logging=True, device='cpu')
    dict_hist["L_"+str(this_L)] = pd.DataFrame(res_dict['hist'])
    # Learn with Tick non-parametric EM (HawkesEM)
    em = HawkesEM((upper-lower), kernel_size=this_L, n_threads=8,
                  verbose=False, tol=1e-3, max_iter=100)
    em.fit([acti_tt_, np.array([]), np.array([])])
    em_baseline = em.baseline
    plt.plot(t, em.get_kernel_values(0, 0, t))
    plt.title("Kernel learned with HawkesEM")
    plt.show()
# %%


# # Learn the estimates using EM in a continuous setting
# _, hist = em_truncated_norm(
#     acti_tt, driver_tt, lower=lower, upper=upper, T=T, sfreq=None,
#     use_dis=False, init_params=init_params, alpha_pos=True,
#     n_iter=max_iter, verbose=False, disable_tqdm=False, compute_loss=True)
# dict_hist["continuous"] = pd.DataFrame(hist)
# # plot learning curves
# plot_hist_params(dict_hist["continuous"], true_params=true_params)
# final_cont_params = dict_hist["continuous"].iloc[-1].to_dict()

# # %% Learn the estimates using a discretized method
# p_lost_list = []
# for this_L in L_list:
#     # project activation on grid defined by this_L
#     driver_tt_ = grid_projection(driver_tt, this_L, remove_duplicates=False)
#     acti_tt_ = grid_projection(acti_tt, this_L, remove_duplicates=False)
#     # _, hist = em_truncated_norm(
#     #     acti_tt_, driver_tt_, lower=lower, upper=upper, T=T, sfreq=this_L,
#     #     use_dis=True, init_params=init_params, alpha_pos=True,
#     #     n_iter=max_iter, verbose=False, disable_tqdm=False, compute_loss=True)
#     # torch solver

#     dict_hist["L_"+str(this_L)] = pd.DataFrame(hist)

# %% Plot, for multiple discretization values, the obtained value of the
# parameter, compared to the one obtained with continuous EM
df_final = pd.DataFrame([dict_hist["L_"+str(this_L)].iloc[-1]
                         for this_L in L_list])
df_final['L'] = L_list
df_final.to_csv('df_final_convergence_estimates.csv')

colors = ['blue', 'orange', 'green']
n_cols = 2
fig, axes = plt.subplots(2, n_cols, figsize=(14, 8), sharex=True)
axes = axes.reshape(-1)
n_axes = len(axes)

# plot baseline
ax = axes[0]
# yy = np.abs(np.array(df_final['baseline']) - final_cont_params['baseline'])
yy = np.abs(np.array(df_final['baseline']) - true_params['baseline'])
ax.plot(L_list, yy, label='baseline', color=colors[0])
# ax.vlines(L, min(yy), max(yy), linestyles='--', colors='black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.set_xlim(min(L_list), max(L_list))
# plot other parameters
for i, param in enumerate(['alpha', 'm', 'sigma']):
    ax = axes[i+1]
    ymin, ymax = np.inf, -np.inf
    # for j in range(len(final_cont_params[param])):
    for j in range(len(true_params[param])):
        # yy = np.abs(np.array([v[j] for v in df_final[param]]
        #                      ) - final_cont_params[param][j])
        yy = np.abs(np.array([v[j] for v in df_final[param]]
                             ) - true_params[param][j])
        ymin, ymax = min(ymin, min(yy)), max(ymax, max(yy))
        ax.plot(L_list, yy, label=f'{param}, kernel {j+1}', color=colors[j])
        if (i+1) in [n_axes-1, n_axes-2]:
            ax.set_xlabel('L')
    # ax.vlines(L, ymin, ymax, linestyles='--', colors='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(L_list), max(L_list))
    ax.legend()

plt.suptitle(f"Absolute estimation error with {solver}")
plt.savefig("fig_convergence_estimates.png")
plt.show()

# if len(p_lost_list) > 0:
#     plt.plot(L_list, p_lost_list)
#     plt.xlim(min(L_list), max(L_list))
#     plt.xscale('log')
#     plt.xlabel('L')
#     plt.ylabel('Percentage (%)')
#     plt.title('Percentage of activation duplicates removed by grid projection.')
#     plt.show()
# %%
