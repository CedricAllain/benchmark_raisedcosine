"""
On MNE sample dataset, compare results obtained with truncated gaussian kernel
(usual DriPP parametrization) and raised cosine kernel (new parametrization)
"""
# %%
import os.path as op
import numpy as np
import pandas as pd
import torch
import ast
from joblib import Memory, Parallel, delayed
from config import SAVE_RESULTS_PATH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
from alphacsc.datasets import somato

from cdl.run_cdl import run_cdl_sample
from cdl import utils
from trunc_norm_kernel.optim import em_truncated_norm
from trunc_norm_kernel.model import TruncNormKernel

from raised_torch.utils.utils import \
    get_sparse_from_tt, grid_projection, check_tensor, get_non_param_estimation
from raised_torch.model import Model
from raised_torch.kernels import raised_cosine_kernel
from raised_torch.solver import initialize, training_loop

# from dripp.experiments.run_multiple_em_on_cdl import \
#     run_multiple_em_on_cdl
# from dripp.config import SAVE_RESULTS_PATH, N_JOBS
# from dripp.trunc_norm_kernel.model import TruncNormKernel
# from dripp.experiments.utils_plot import plot_cdl_atoms

# from mne.time_frequency import tfr_morlet


# run CDL on Somato
cdl_params = {
    'n_atoms': 40,
    'sfreq': 150.,
    'n_iter': 100,
    'eps': 1e-4,
    'reg': 0.1,
    'n_jobs': 5,
    'n_splits': 10
}
dict_global = run_cdl_sample(**cdl_params)
sfreq = dict_global['dict_other_params']['sfreq']
dict_pair_up = dict_global['dict_pair_up']


u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])
v_hat_ = np.array(dict_global['dict_cdl_fit_res']['v_hat_'])

T = dict_pair_up['T']
events_timestamps = dict_pair_up['events_timestamps']  # events timestamps
acti = np.array(dict_pair_up['acti_shift'])  # atoms' activations
acti = utils.filter_activation(
    acti, atom_to_filter='all', sfreq=sfreq, time_interval=0.01)
atoms_timestamps = utils.get_atoms_timestamps(
    acti=acti, sfreq=sfreq, threshold=0.6e-10)

# prepare combination to run parallel EM
list_tasks = ([1, 2], [3, 4])
n_atoms = dict_global['dict_cdl_params']['n_atoms']
list_atoms = list(range(n_atoms))
combs_atoms_tasks = [(kk, list_tasks) for kk in list_atoms]

# run CDL and EM
lower, upper = 30e-3, 500e-3
shift_acti = True
n_iter = 400

em_params = {'events_timestamps': events_timestamps,
             'atoms_timestamps': atoms_timestamps,
             'lower': lower, 'upper': upper,
             'T': T, 'initializer': 'smart_start',
             'alpha_pos': True,
             'n_iter': n_iter}
# %%


def procedure_em(comb):
    """Procedure to parallelized.

    Parameters
    ----------
    comb : tuple
        tuple ((atom, task), args) on which to perform the EM algorithm
        where,
            atom : int, the atom idex
            task : int | array-like, ids of tasks
            args : dict, dictionary of EM parameters, with following keys
                lower, upper : int | float
                T : int | float
                initializer : str
                early_stopping : str | None
                early_stopping_params : dict | None
                alpha_pos : bool
                n_iter : int | array-like
                    if array-like, returns the value of learned parameters at
                    the different values of n_iter

    Return
    ------
    new_row : dict | list of dict
        new row(s) of the results DataFrame
        return a list of dict if n_iter's type is array-like

    """
    atom, tasks = comb

    n_iter = em_params['n_iter']

    # get activation timestamps
    atoms_timestamps = np.array(em_params['atoms_timestamps'])
    aa = atoms_timestamps[atom]

    # get and merge tasks timestamps
    events_timestamps = em_params['events_timestamps']  # dict

    def proprocess_tasks(tasks):
        if isinstance(tasks, int):
            tt = np.sort(events_timestamps[tasks])
        elif isinstance(tasks, list):
            tt = np.r_[events_timestamps[tasks[0]]]
            for i in tasks[1:]:
                tt = np.r_[tt, events_timestamps[i]]
            tt = np.sort(tt)

        return tt

    if isinstance(tasks, (tuple, list)):
        # in that case, multiple drivers
        tt = np.array([proprocess_tasks(task) for task in tasks])
    else:
        tt = proprocess_tasks(tasks)

    # base row
    base_row = {'atom': int(atom),
                'tasks': tasks,
                'lower': em_params['lower'],
                'upper': em_params['upper'],
                'initializer': em_params['initializer']}

    # run EM algorithm
    res_em = em_truncated_norm(
        acti_tt=aa,
        driver_tt=tt,
        lower=em_params['lower'],
        upper=em_params['upper'],
        T=em_params['T'],
        initializer=em_params['initializer'],
        alpha_pos=em_params['alpha_pos'],
        n_iter=n_iter,
        verbose=True,
        disable_tqdm=True)

    # get results
    res_params, history_params = res_em

    baseline_hat, alpha_hat, m_hat, sigma_hat = res_params
    new_row = {**base_row,
               'n_iter': n_iter,
               'baseline_hat': baseline_hat,
               'alpha_hat': alpha_hat,
               'm_hat': m_hat,
               'sigma_hat': sigma_hat,
               'baseline_init': history_params[0]['baseline'],
               'alpha_init': history_params[0]['alpha']}

    return [new_row]


if (SAVE_RESULTS_PATH / 'df_res_sample_em.csv').exists():
    df_res_em = pd.read_pickle(SAVE_RESULTS_PATH / 'df_res_sample_em.csv')
else:
    df_res_em = pd.DataFrame()
    new_rows = Parallel(n_jobs=10, verbose=1)(
        delayed(procedure_em)(this_comb) for this_comb in combs_atoms_tasks)

    for new_row in new_rows:
        df_res_em = df_res_em.append(new_row, ignore_index=True)

    path_df_res = SAVE_RESULTS_PATH
    if not path_df_res.exists():
        path_df_res.mkdir(parents=True)

    df_res_em.to_pickle(SAVE_RESULTS_PATH / 'df_res_sample_em.csv')

# %% get results with raised kernel
kernel_name = 'raised_cosine'
loss_name = 'log-likelihood'
L = 100
dt = 1 / L
t = torch.arange(lower, upper+dt, dt)
recompute = True
# get sparse driver vectors


def proprocess_tasks(tasks):
    if isinstance(tasks, int):
        tt = np.sort(events_timestamps[tasks])
    elif isinstance(tasks, list):
        tt = np.r_[events_timestamps[tasks[0]]]
        for i in tasks[1:]:
            tt = np.r_[tt, events_timestamps[i]]
        tt = np.sort(tt)

    return tt


driver_tt = [proprocess_tasks(tasks) for tasks in list_tasks]

driver = get_sparse_from_tt(driver_tt, T, dt)
acti_tt = grid_projection(
    atoms_timestamps, L, remove_duplicates=False, verbose=False)
acti = get_sparse_from_tt(acti_tt, T, dt)


def procedure_torch(kk):
    init_params = initialize(driver_tt=driver_tt,
                             acti_tt=acti_tt[kk],
                             T=T, initializer='smart_start',
                             lower=lower, upper=upper,
                             kernel_name=kernel_name)
    baseline_init, alpha_init, m_init, sigma_init = init_params
    model = Model(t, baseline_init, alpha_init, m_init, sigma_init,
                  dt=dt, kernel_name=kernel_name, loss_name=loss_name,
                  lower=lower, upper=upper, driver=driver)

    res_dict = training_loop(model, driver, acti[kk], T, solver='RMSprop',
                             step_size=1e-3, max_iter=100, test=False,
                             logging=True, device='cpu')

    base_row = {'atom': int(kk),
                'lower': em_params['lower'],
                'upper': em_params['upper'],
                'initializer': em_params['initializer']}

    new_row = {**base_row,
               'n_iter': n_iter,
               'baseline_hat': np.array(res_dict['est_params']['baseline']),
               'alpha_hat': np.array(res_dict['est_params']['alpha']),
               'm_hat': np.array(res_dict['est_params']['m']),
               'sigma_hat': np.array(res_dict['est_params']['sigma']),
               'baseline_init': baseline_init,
               'alpha_init': alpha_init}

    return [new_row]


if (SAVE_RESULTS_PATH / 'df_res_sample_torch.csv').exists() and not recompute:
    df_res_torch = pd.read_pickle(
        SAVE_RESULTS_PATH / 'df_res_sample_torch.csv')
    list_cols = ['alpha_hat', 'sigma_hat', 'm_hat', 'alpha_init']
    for col in list_cols:
        df_res_torch[col] = df_res_torch[col].apply(ast.literal_eval)
else:
    df_res_torch = pd.DataFrame()
    new_rows = Parallel(n_jobs=10, verbose=1)(
        delayed(procedure_torch)(kk) for kk in range(cdl_params['n_atoms']))

    for new_row in new_rows:
        df_res_torch = df_res_torch.append(new_row, ignore_index=True)

    path_df_res = SAVE_RESULTS_PATH
    if not path_df_res.exists():
        path_df_res.mkdir(parents=True)

    df_res_torch.to_pickle(SAVE_RESULTS_PATH / 'df_res_sample_torch.csv')
# %% Plot results
data_utils = utils.get_data_utils(data_source='sample', verbose=False)
raw = mne.io.read_raw_fif(data_utils['file_name'], preload=True)
raw.pick_types(meg='grad', eeg=False, eog=True, stim=True)
raw.notch_filter(np.arange(60, 181, 60))
raw.filter(l_freq=2, h_freq=None)

# get info only for MEG
info = raw.copy().pick_types(meg=True).info

# %%

plotted_atoms = [0, 1, 2, 6]
# plotted_atoms = [0, 1, 6]
plotted_tasks = {'auditory': [1, 2],
                 'visual': [3, 4]}

fontsize = 8
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "xtick.labelsize": fontsize,
    'ytick.labelsize': fontsize,
})

colors = ['blue', 'orange']

# fig = plt.figure(figsize=(5.5, 4.7))
# gs = gridspec.GridSpec(nrows=4, ncols=4, hspace=0.26, wspace=0.18, figure=fig)
fig = plt.figure(figsize=(6.5, 6))
gs = gridspec.GridSpec(nrows=5, ncols=4, hspace=0.26, wspace=0.35, figure=fig)

# x axis for temporal pattern
n_times_atom = dict_global['dict_cdl_params']['n_times_atom']
t = np.arange(n_times_atom) / cdl_params['sfreq']
# x axis for estimated intensity function
xx = np.linspace(0, 500e-3, 500)

for ii, kk in enumerate(plotted_atoms):
    # Select the current atom
    u_k = u_hat_[kk]
    v_k = v_hat_[kk]

    # Plot the spatial map of the atom using mne topomap
    ax = fig.add_subplot(gs[0, ii])
    mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
    ax.set_title('Atom % d' % kk, fontsize=fontsize, pad=0)
    if ii == 0:
        ax.set_ylabel('Spatial', labelpad=28, fontsize=fontsize)

    # Plot the temporal pattern of the atom
    ax = fig.add_subplot(gs[1, ii])
    if kk != 0:
        v_k = -1 * np.array(v_k)
    ax.plot(t, v_k)

    if ii == 0:
        temporal_ax = ax
        ax.set_ylabel('Temporal', fontsize=fontsize)

    if ii > 0:
        ax.get_yaxis().set_visible(False)
        temporal_ax.get_shared_y_axes().join(temporal_ax, ax)
        ax.autoscale()

    ax.set_xlim(0, 1)
    ax.set_xticklabels([0, 0.5, 1], fontsize=fontsize)

    # Plot the learned density kernel
    ax = fig.add_subplot(gs[2, ii])

    has_m_line = False
    df_temp = df_res_em[(df_res_em['atom'] == kk)]
    for jj, label in enumerate(plotted_tasks.keys()):
        # select sub-df of interest
        # in case that there has been an early stopping
        n_iter_temp = min(n_iter, df_temp['n_iter'].values.max())
        df_temp = df_temp[df_temp['n_iter'] == n_iter_temp]
        # unpack parameters estimates
        alpha = list(df_temp['alpha_hat'])[0][jj]
        baseline = list(df_temp['baseline_hat'])[0]
        m = list(df_temp['m_hat'])[0][jj]
        sigma = list(df_temp['sigma_hat'])[0][jj]

        # define kernel function
        kernel = TruncNormKernel(lower, upper, m, sigma)
        yy = baseline + alpha * kernel.eval(xx)
        lambda_max = baseline + alpha * kernel.max
        # ratio_lambda_max = lambda_max / baseline
        ratio_lambda_max = alpha / baseline

        if ii > 0:
            plot_label = None
        else:
            plot_label = label

        ax.plot(xx, yy, color=colors[jj], label=plot_label)

        if (ratio_lambda_max > 1) and kk not in [0, 1]:
            has_m_line = True
            ax.vlines(m, ymin=0, ymax=lambda_max, color='black',
                      linestyle='--', label=r'%.3f' % m)

    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_xticklabels([0, 0.25, 0.5], fontsize=fontsize)

    if ii == 0:
        intensity_ax = ax
        ax.set_ylabel('Intensity TG', labelpad=7, fontsize=fontsize)
    else:
        ax.get_yaxis().set_visible(False)
        intensity_ax.get_shared_y_axes().join(intensity_ax, ax)
        ax.autoscale()

    ax.set_xlim(0, 500e-3)

    if plot_label is not None or has_m_line:
        ax.legend(fontsize=fontsize, handlelength=1)

    # Plot the learned density kernel with torch
    ax = fig.add_subplot(gs[3, ii])

    # get torch results
    df_temp_torch = df_res_torch[(df_res_torch['atom'] == kk)]
    baseline_torch = check_tensor(df_temp_torch['baseline_hat'].iloc[0])
    alpha = df_temp_torch['alpha_hat'].iloc[0]
    m = df_temp_torch['m_hat'].iloc[0]
    sigma = df_temp_torch['sigma_hat'].iloc[0]
    kernel_torch = raised_cosine_kernel(t=xx, alpha=alpha, u=m, sigma=sigma)

    has_m_line = False
    plot_label = None
    for jj, this_kernel in enumerate(kernel_torch):
        # if ii > 0:
        #     plot_label = None
        # else:
        #     plot_label = label

        ax.plot(xx, baseline_torch + this_kernel,
                color=colors[jj])

        if (alpha[jj]/baseline_torch > 1) and kk not in [0, 1]:
            has_m_line = True
            lambda_max = baseline_torch + max(this_kernel)
            ax.vlines(m[jj]+sigma[jj], ymin=0, ymax=lambda_max, color='black',
                      linestyle='--', label=r'%.3f' % (m[jj]+sigma[jj]))

    if ii == 0:
        intensity_ax_rc = ax
        ax.set_ylabel('Intensity RC', labelpad=7, fontsize=fontsize)
    elif kk != 2:
        ax.get_yaxis().set_visible(False)
        intensity_ax_rc.get_shared_y_axes().join(intensity_ax_rc, ax)
        ax.autoscale()

    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_xticklabels([0, 0.25, 0.5], fontsize=fontsize)

    ax.set_xlim(0, 500e-3)

    if plot_label is not None or has_m_line:
        ax.legend(fontsize=fontsize, handlelength=1)

    # Fit non param model
    kernel_support = 0.5
    em = get_non_param_estimation(
        kernel_support=kernel_support, kernel_size=int(L*kernel_support),
        acti_tt=acti_tt[kk], driver_tt=driver_tt)
    baseline_np = em.baseline[0]
    t_np = np.arange(0, 0.5, 1/em.kernel_size)

    ax = fig.add_subplot(gs[4, ii])

    max_np = 0
    m_np = 0
    for jj in [1, 2]:
        yy = baseline_np + em.get_kernel_values(0, jj, t_np)
        ax.plot(t_np, yy, color=colors[jj-1])
        if max(yy) > max_np:
            max_np = max(yy)
            m_np = np.argmax(yy) / em.kernel_size

    if kk not in [0, 1]:
        lambda_max = baseline_np + max_np
        ax.vlines(m_np, ymin=0, ymax=lambda_max, color='black',
                  linestyle='--', label=r'%.3f' % (m_np))
        ax.legend(fontsize=fontsize, handlelength=1)

    if ii == 0:
        intensity_ax_np = ax
        ax.set_ylabel('Intensity NP', labelpad=7, fontsize=fontsize)
    else:
        ax.get_yaxis().set_visible(False)
        intensity_ax_rc.get_shared_y_axes().join(intensity_ax_np, ax)
        ax.autoscale()

    ax.set_xlim(0, 500e-3)


# save figure
path_fig = SAVE_RESULTS_PATH / 'fig4.pdf'
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
plt.savefig(str(path_fig).replace('pdf', 'png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %%
