"""
Plot, as a function of the grid discretization step, the convergence of final
estimates compared to the estimates using EM in a continuous setting.
"""

# %%
# %%
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
from joblib import Memory, Parallel, delayed
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

CACHEDIR = Path('./__cache__')
memory = Memory(CACHEDIR, verbose=0)


# %% true parameters
# n_drivers = 2
# baseline = 1
# alpha = np.array([1.])
# m = np.array([0.4])
# sigma = np.array([0.1])
# lower, upper = 0, 1
# true_params = {'baseline': baseline, 'alpha': alpha, 'm': m, 'sigma': sigma}
# max_iter = 200

# # %% simulation parameters on a continuous line (both driver and activation tt)
# T = 1_000
# p_task = 0.4
# isi = [1.2, 1.2]
# seed = 42
# driver_tt, acti_tt, kernel, intensity = simulate_data(
#     lower=lower, upper=upper, m=m, sigma=sigma, sfreq=None, baseline=baseline,
#     alpha=alpha, T=T, isi=isi, add_jitter=True, n_tasks=p_task,
#     n_drivers=n_drivers, seed=seed, return_nll=False, verbose=True)

# # %% initialize parameters,
# kernel_name = 'gaussian'
# init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
#                          lower=lower, upper=upper,
#                          kernel_name=kernel_name)
# baseline_init, alpha_init, m_init, sigma_init = init_params
# init_params[0] = [baseline_init, np.nan]

# # %% Learn the estimates using EM in a continuous setting
# res_params, hist = em_truncated_norm(
#     acti_tt, driver_tt, lower=lower, upper=upper, T=T, sfreq=None,
#     use_dis=False, init_params=init_params, alpha_pos=True,
#     n_iter=max_iter, verbose=False, disable_tqdm=False, compute_loss=True)
# # plot learning curves
# plot_hist_params(pd.DataFrame(hist), true_params=true_params)
# cont_params = pd.DataFrame(hist).iloc[-1].to_dict()

# %% varying discretization step, learn params with torch model


lower, upper = 0, 1
max_iter = 50


@memory.cache(ignore=['driver_tt', 'acti_tt', 'init_params', 'cont_params'])
def compute_discretization_error(T, L, seed, driver_tt, acti_tt, init_params,
                                 true_params, cont_params):
    """

    """

    # grid projection of timestamps
    driver_tt_ = grid_projection(
        driver_tt, L, remove_duplicates=False)
    acti_tt_ = grid_projection(acti_tt, L, remove_duplicates=False)

    # learn with EM
    res_params, hist = em_truncated_norm(
        acti_tt_, driver_tt_, lower=lower, upper=upper, T=T, sfreq=L,
        use_dis=True, init_params=init_params, alpha_pos=True,
        n_iter=max_iter, verbose=False, disable_tqdm=True, compute_loss=True)
    baseline_em, alpha_em, m_em, sigma_em = res_params
    params_em = {'baseline': baseline_em, 'alpha': alpha_em,
                 'm': np.array(m_em), 'sigma': np.array(sigma_em)}
    time_em = hist[-1]['time_loop']

    em_err_cont = {k + '_em_err_cont': np.abs(params_em[k] - cont_params[k])
                   for k in params_em.keys()}
    em_err_true = {k + '_em_err_true': np.abs(params_em[k] - true_params[k])
                   for k in params_em.keys()}

    # learn with Torch
    # dt = 1/L
    # t = torch.arange(lower, upper, dt)  # maximal kernel support
    # # make sparse vector from timestamps
    # driver = get_sparse_from_tt(driver_tt_, T, dt)
    # acti = get_sparse_from_tt(acti_tt_, T, dt)
    # baseline_init, alpha_init, m_init, sigma_init = init_params
    # model = Model(t, baseline_init, alpha_init, m_init, sigma_init,
    #               dt, 'gaussian', 'log-likelihood', lower, upper, driver)
    # res_dict = training_loop(model, driver, acti, T=T, solver='RMSprop',
    #                          step_size=1e-3, max_iter=100, test=False,
    #                          logging=True, device='cpu')
    # time_torch = res_dict['compute_time']
    # params_torch = {k: np.array(v) for k, v in res_dict['est_params'].items()}

    # torch_err_cont = {k + '_torch_err_cont': np.abs(params_torch[k] - cont_params[k])
    #                   for k in params_torch.keys()}
    # torch_err_true = {k + '_torch_err_true': np.abs(params_torch[k] - true_params[k])
    #                   for k in params_torch.keys()}

    # save results
    this_row = {'T': T, 'L': L, 'seed': seed,
                'time_em': time_em,  # 'time_torch': time_torch,
                # 'params_em': params_em, 'params_torch': params_torch,
                **em_err_cont, **em_err_true}
    # **torch_err_cont, **torch_err_true}

    return this_row


def procedure(T, seed):

    # define exeperiment parameters
    n_drivers = 1
    baseline = 1
    alpha = np.array([1.])
    m = np.array([0.4])
    sigma = np.array([0.1])
    lower, upper = 0, 1
    true_params = {'baseline': baseline,
                   'alpha': alpha, 'm': m, 'sigma': sigma}

    # simulation parameters on a continuous line (both driver and activation tt)
    driver_tt, acti_tt, kernel, intensity = simulate_data(
        lower=lower, upper=upper, m=m, sigma=sigma, sfreq=None,
        baseline=baseline,
        alpha=alpha, T=T, n_drivers=n_drivers, seed=seed, return_nll=False, verbose=True,
        poisson_intensity=0.5)

    # initialize parameters,
    init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
                             lower=lower, upper=upper,
                             kernel_name='gaussian')
    baseline_init, alpha_init, m_init, sigma_init = init_params
    # init_params[0] = [baseline_init, np.nan]

    # Learn the estimates using EM in a continuous setting
    _, hist = em_truncated_norm(
        acti_tt, driver_tt, lower=lower, upper=upper, T=T, sfreq=None,
        use_dis=False, init_params=init_params, alpha_pos=True,
        n_iter=max_iter, verbose=False, disable_tqdm=False, compute_loss=True)
    # plot learning curves
    # plot_hist_params(pd.DataFrame(hist), true_params=true_params)
    cont_params = pd.DataFrame(hist).iloc[-1].to_dict()

    # fit models and compute error
    # L_list = [10, 20, 30, 40, 50, 60, 70, 80, 90,
    #           100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    L_list = [i*10**j for j in [1, 2, 3] for i in range(1, 10)]
    L_list.append(10**4)
    rows = []
    for this_L in tqdm(L_list):
        this_row = compute_discretization_error(
            T, this_L, seed, driver_tt, acti_tt, init_params, true_params,
            cont_params)
        rows.append(this_row)

    df = pd.DataFrame(rows)

    return df


# %%

# L_list = [i*10**j for j in [1, 2, 3] for i in range(1, 10)]
# L_list.append(10**4)

df = pd.DataFrame()
list_seed = list(range(30))
for this_T in [1_000, 2_000]:
    print("T =", this_T)
    new_dfs = Parallel(n_jobs=min(30, len(list_seed)), verbose=1)(
        delayed(procedure)(this_T, this_seed) for this_seed in list_seed)
    new_dfs.append(df)
    df = pd.concat(new_dfs)
    df.to_pickle('df_convergence_estimates_em.csv')


# %% Plot, for multiple discretization values, the obtained value of the

df = pd.read_pickle('df_convergence_estimates.csv')

for param in ['alpha', 'm', 'sigma']:
    # for pre in ['em', 'torch']:
    pre = 'em'
    for suf in ['true', 'cont']:
        col = (param + '_' + pre + '_err_' + suf)
        df[col] = df[col].apply(lambda x: x[0])

L_list = df['L'].unique()


# for pre in ['em', 'torch']:
for pre in ['em']:
    for suf in ['true', 'cont']:

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.reshape(-1)

        for i, param in enumerate(['baseline', 'alpha', 'm', 'sigma']):
            ax = axes[i]
            col = (param + '_' + pre + '_err_' + suf)
            sns.lineplot(data=df, x="L", y=col, hue="T", ax=ax)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(min(L_list), max(L_list))

        plt.suptitle(
            f"Absolute estimation error between {pre} estimates and {suf} parameters")
        plt.savefig("fig_convergence_estimates" + pre + "_" + suf + ".png")
        plt.show()


# %%
