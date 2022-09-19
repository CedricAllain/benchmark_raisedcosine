"""
On MNE somato dataset, compare results obtained with truncated gaussian kernel
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

from cdl.run_cdl import run_cdl_somato
from cdl import utils
from trunc_norm_kernel.optim import em_truncated_norm
from trunc_norm_kernel.model import TruncNormKernel

from raised_torch.utils.utils import get_sparse_from_tt, grid_projection, \
    check_tensor, get_non_param_estimation
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
    'sfreq': 150.,
    'n_iter': 100,
    'eps': 1e-4,
    'n_jobs': 5,
    'n_splits': 10,
    'n_atoms': 20,
    'n_times_atom': 80,
    'reg': 0.2
}
dict_global = run_cdl_somato(**cdl_params)
sfreq = dict_global['dict_other_params']['sfreq']
dict_pair_up = dict_global['dict_pair_up']

T = dict_pair_up['T']
events_timestamps = dict_pair_up['events_timestamps']  # events timestamps
acti = np.array(dict_pair_up['acti_shift'])  # atoms' activations
acti = utils.filter_activation(
    acti, atom_to_filter='all', sfreq=sfreq, time_interval=0.01)
atoms_timestamps = utils.get_atoms_timestamps(
    acti=acti, sfreq=sfreq, threshold=1e-10)


# prepare combination to run parallel EM
list_tasks = [1]
n_atoms = dict_global['dict_cdl_params']['n_atoms']
list_atoms = list(range(n_atoms))
combs_atoms_tasks = [(kk, list_tasks) for kk in list_atoms]

# run CDL and EM
lower, upper = 0, 2
shift_acti = True
threshold = 1e-10
n_iter = 400

em_params = {'events_timestamps': events_timestamps,
             'atoms_timestamps': atoms_timestamps,
             'lower': lower, 'upper': upper,
             'T': T, 'initializer': 'smart_start',
             'alpha_pos': True,
             'n_iter': 400}

# %%
recompute = True


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


if (SAVE_RESULTS_PATH / 'df_res_em.csv').exists() and not recompute:
    df_res_em = pd.read_pickle(SAVE_RESULTS_PATH / 'df_res_em.csv')
    # list_cols = ['tasks', 'alpha_hat', 'sigma_hat', 'm_hat', 'alpha_init']
    # for col in list_cols:
    #     df_res_em[col] = df_res_em[col].apply(ast.literal_eval)
else:
    df_res_em = pd.DataFrame()
    new_rows = Parallel(n_jobs=10, verbose=1)(
        delayed(procedure_em)(this_comb) for this_comb in combs_atoms_tasks)

    for new_row in new_rows:
        df_res_em = df_res_em.append(new_row, ignore_index=True)

    path_df_res = SAVE_RESULTS_PATH
    if not path_df_res.exists():
        path_df_res.mkdir(parents=True)

    df_res_em.to_pickle(SAVE_RESULTS_PATH / 'df_res_em.csv')

# %% get results with raised kernel
kernel_name = 'raised_cosine'
loss_name = 'log-likelihood'
L = 100
dt = 1 / L
t = torch.arange(lower, upper+dt, dt)
recompute = True

# get sparse driver vectors
driver = get_sparse_from_tt([events_timestamps[1]], T, dt)
acti_tt = grid_projection(
    atoms_timestamps, L, remove_duplicates=False, verbose=False)
acti = get_sparse_from_tt(acti_tt, T, dt)


def procedure_torch(kk):
    init_params = initialize(driver_tt=[events_timestamps[1]],
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


if (SAVE_RESULTS_PATH / 'df_res_torch.csv').exists() and not recompute:
    df_res_torch = pd.read_pickle(SAVE_RESULTS_PATH / 'df_res_torch.csv')
    # list_cols = ['alpha_hat', 'sigma_hat', 'm_hat', 'alpha_init']
    # for col in list_cols:
    #     df_res_torch[col] = df_res_torch[col].apply(ast.literal_eval)
else:
    df_res_torch = pd.DataFrame()
    new_rows = Parallel(n_jobs=10, verbose=1)(
        delayed(procedure_torch)(kk) for kk in range(cdl_params['n_atoms']))

    for new_row in new_rows:
        df_res_torch = df_res_torch.append(new_row, ignore_index=True)

    path_df_res = SAVE_RESULTS_PATH
    if not path_df_res.exists():
        path_df_res.mkdir(parents=True)

    df_res_torch.to_pickle(SAVE_RESULTS_PATH / 'df_res_torch.csv')
# %% Plot results

sfreq = cdl_params['sfreq']
_, info = somato.load_data(sfreq=sfreq)

plotted_atoms_list = [[0, 2, 7], [2, 7, 10], [1, 2, 4], [0, 7, 10]]

fontsize = 8
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "xtick.labelsize": fontsize,
    'ytick.labelsize': fontsize,
    'legend.title_fontsize': fontsize
})

colors = ['blue', 'green', 'orange']

n_times_atom = cdl_params['n_times_atom']


u_hat_ = np.array(dict_global['dict_cdl_fit_res']['u_hat_'])
v_hat_ = np.array(dict_global['dict_cdl_fit_res']['v_hat_'])

# x axis for temporal pattern
t = np.arange(n_times_atom) / sfreq
# x axis for estimate intensity
xx = np.linspace(0, 2, 500)

for plotted_atoms in plotted_atoms_list:
    # define figure
    fig = plt.figure(figsize=(5.5, 3.5 / 3 * 2))
    ratio = 1.5  # ratio between width of atom plot and intensity plot
    step = 1/(3+ratio)
    gs = gridspec.GridSpec(nrows=2, ncols=4,
                           width_ratios=[step, step, step, ratio*step],
                           hspace=0.05,
                           wspace=0.1,
                           figure=fig)

    # plot spatial and temporal pattern
    for ii, kk in enumerate(plotted_atoms):
        # Select the current atom
        u_k = u_hat_[kk]
        v_k = v_hat_[kk]

        # plot spatial pattern
        ax = fig.add_subplot(gs[0, ii])
        ax.set_title('Atom % d' % kk, fontsize=fontsize)
        mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
        if ii == 0:
            ax.set_ylabel('Spatial', labelpad=32, fontsize=fontsize)

        # plot temporal pattern
        ax = fig.add_subplot(gs[1, ii])

        if kk == 0:  # return atom 0
            v_k = -1 * np.array(v_k)

        ax.plot(t, v_k, color=colors[ii])
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        if ii == 0:
            first_ax = ax
            ax.set_ylabel('Temporal', fontsize=fontsize)
        else:
            ax.get_yaxis().set_visible(False)
            first_ax.get_shared_y_axes().join(first_ax, ax)
            ax.autoscale()

        ax.set_xlim(0, n_times_atom / sfreq)
        ax.set_xticks([0, 0.25, 0.5])
        ax.set_xticklabels([0, 0.25, 0.5], fontsize=fontsize)

    # plot intensities
    ax = fig.add_subplot(gs[:, -1:])
    ax.set_title('Intensity', fontsize=fontsize)
    for ii, kk in enumerate(plotted_atoms):
        # # plot EM-learned intensities
        # # select sub-df of interest
        # df_temp = df_res_em[(df_res_em['atom'] == kk)
        #                     & (df_res_em['lower'] == lower)
        #                     & (df_res_em['upper'] == upper)]

        # # if we save several values for n_iter
        # if df_temp.shape[0] != 1:
        #     # in case that there has been an early stopping
        #     n_iter_temp = min(
        #         n_iter, df_temp['n_iter'].values.max())
        #     df_temp = df_temp[df_temp['n_iter'] == n_iter_temp]

        # list_yy = []
        # for i in df_temp.index:
        #     # unpack parameters estimates
        #     alpha = df_temp['alpha_hat'][i][0]
        #     baseline = df_temp['baseline_hat'][i]
        #     m = df_temp['m_hat'][i][0]
        #     sigma = df_temp['sigma_hat'][i][0]

        #     # define kernel function
        #     kernel = TruncNormKernel(lower, upper, m, sigma)
        #     yy = baseline + alpha * kernel.eval(xx)
        #     list_yy.append(yy)

        # # plot torch
        # df_temp_torch = df_res_torch[(df_res_torch['atom'] == kk)]
        # baseline = check_tensor(df_temp_torch['baseline_hat'].iloc[0])
        # alpha = df_temp_torch['alpha_hat'].iloc[0]
        # m = df_temp_torch['m_hat'].iloc[0]
        # sigma = df_temp_torch['sigma_hat'].iloc[0]
        # kernel = raised_cosine_kernel(t=xx, alpha=alpha, u=m, sigma=sigma)

        # label = '% d' % kk
        # ax.plot(xx, yy, label=label, color=colors[ii])
        # ax.plot(xx, baseline + kernel[0], color=colors[ii], linestyle='--')

        # Learn non parametric
        em = get_non_param_estimation(
            kernel_support=2, kernel_size=L*2, acti_tt=acti_tt[kk],
            driver_tt=[events_timestamps[1]])
        baseline_np = em.baseline[0]
        ax.plot(xx, baseline_np + em.get_kernel_values(0, 1, xx),
                color=colors[ii], linestyle='-', alpha=0.6)

        ax.set_xlim(0, 2)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.yaxis.set_ticks_position("right")
        # ax.set_yscale('log')
        ax.legend(fontsize=fontsize, handlelength=1, title='Atom')

    # save figure
    suffix = 'atom'
    for kk in plotted_atoms:
        suffix += '_' + str(kk)
    name = 'fig5_' + suffix + '_bis.pdf'
    path_fig = SAVE_RESULTS_PATH / name
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.savefig(str(path_fig).replace('pdf', 'png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %%
