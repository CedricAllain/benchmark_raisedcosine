"""
Plot, as a function of the grid discretization step, the convergence of final
estimates compared to the estimates using EM in a continuous setting.
"""

# %%
# %%

from trunc_norm_kernel.metric import negative_log_likelihood
from trunc_norm_kernel.model import TruncNormKernel, Intensity
from trunc_norm_kernel.optim import em_truncated_norm
from trunc_norm_kernel.simu import simulate_data
from raised_torch.utils.utils_plot import plot_hist_params
from raised_torch.utils.utils import grid_projection, check_tensor, get_sparse_from_tt
from raised_torch.kernels import compute_kernels
from raised_torch.solver import initialize, training_loop, compute_loss, optimizer
from raised_torch.model import Model
from raised_torch.simu_pp import simu
from tick.hawkes import HawkesEM
from ast import increment_lineno
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import json
import torch
from tqdm import tqdm
from joblib import Memory, Parallel, delayed, hash
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from tueplots import bundles
from tueplots import figsizes
# plt.rcParams.update(bundles.iclr2023())
# plt.rcParams.update({"figure.dpi": 300})

FONTSIZE = 11
# plt.rcParams["figure.figsize"] = (5, 3.2)
# plt.rcParams["axes.grid"] = False
# plt.rcParams["axes.grid.axis"] = "y"
# plt.rcParams["grid.linestyle"] = "--"
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = '\\renewcommand{\\rmdefault}{ptm}\\renewcommand{\\sfdefault}{phv}'

plt.rc('legend', fontsize=FONTSIZE-1)


CACHEDIR = Path('./__cache__')
memory = Memory(CACHEDIR, verbose=0)

SAVE_RESULTS_PATH = Path('./fig1')
if not SAVE_RESULTS_PATH.exists():
    SAVE_RESULTS_PATH.mkdir(parents=True)


# @memory.cache(ignore=['driver_tt', 'acti_tt', 'init_params', 'cont_params'])
def compute_discretization_error(T, L, seed, driver_tt, acti_tt, init_params,
                                 true_params, cont_params, poisson_intensity):
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
    this_row = {'T': T, 'L': L, 'seed': seed, 'poisson_intensity': poisson_intensity,
                'time_em': time_em,  # 'time_torch': time_torch,
                # 'params_em': params_em, 'params_torch': params_torch,
                **{k+'_em': v for k, v in params_em.items()},
                **{k+'_true': v for k, v in true_params.items()},
                **{k+'_cont': v for k, v in cont_params.items()},
                **em_err_cont, **em_err_true}
    # **torch_err_cont, **torch_err_true}

    return this_row


def procedure(true_params, poisson_intensity, T, seed):

    # simulation parameters on a continuous line (both driver and activation tt)
    driver_tt, acti_tt, kernel, intensity = simulate_data(
        lower=lower, upper=upper,
        m=true_params['m'], sigma=true_params['sigma'],
        sfreq=None,
        baseline=true_params['baseline'], alpha=true_params['alpha'],
        T=T, n_drivers=n_drivers, seed=seed,
        return_nll=False, verbose=True, poisson_intensity=poisson_intensity)

    # initialize parameters,
    init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
                             lower=lower, upper=upper,
                             kernel_name='gaussian')

    # Learn the estimates using EM in a continuous setting
    _, hist = em_truncated_norm(
        acti_tt, driver_tt, lower=lower, upper=upper, T=T, sfreq=None,
        use_dis=False, init_params=init_params, alpha_pos=True,
        n_iter=max_iter, verbose=False, disable_tqdm=False, compute_loss=True)
    # plot learning curves
    # plot_hist_params(pd.DataFrame(hist), true_params=true_params)
    cont_params = pd.DataFrame(hist).iloc[-1].to_dict()

    # fit models and compute error
    # L_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    L_list = [i*10**j for j in [1, 2, 3] for i in range(1, 10)]
    L_list.append(10**4)
    rows = []
    for this_L in tqdm(L_list):
        this_row = compute_discretization_error(
            T, this_L, seed, driver_tt, acti_tt, init_params, true_params,
            cont_params, poisson_intensity)
        rows.append(this_row)

    df = pd.DataFrame(rows)

    return df


dict_name_latex = {'baseline': r'$\mu$',
                   'alpha': r'$\alpha$',
                   'm': r'$m$',
                   'sigma': r'$\sigma$'}


def plot_fig1(folder_name, L_max=10**3):
    """

    """

    df = pd.read_pickle(folder_name / 'df_convergence_estimates_em.csv')

    pre = 'em'
    cols = [param + '_' + pre + '_err_' + suf
            for param in ['alpha', 'm', 'sigma']
            for suf in ['true', 'cont']]

    params = ['alpha', 'm', 'sigma']
    cols += [p+suf for p in params for suf in ['_cont', '_true', '_em']]

    for col in cols:
        print(col)
        df[col] = df[col].apply(lambda x: x[0])

    for param in ['baseline', 'alpha', 'm', 'sigma']:
        df[param +
            '_cont_err_true'] = np.abs(df[param + '_cont'] - df[param + '_true'])

    df['dt'] = 1 / df['L']

    sub_df = df[df['L'] <= L_max]

    cols_em = [param + '_em_err_true'
               for param in ['baseline', 'alpha', 'm', 'sigma']]
    sub_df_em = sub_df[cols_em + ['T', 'L', 'dt', 'seed']]
    sub_df_em['estimates'] = 'EM'
    sub_df_em.rename(columns={col: col.replace('_em', '') for col in cols_em},
                     inplace=True)

    cols_cont = [param + '_cont_err_true'
                 for param in ['baseline', 'alpha', 'm', 'sigma']]
    sub_df_cont = sub_df[cols_cont + ['T', 'L', 'dt', 'seed']]
    sub_df_cont['estimates'] = 'continuous'
    sub_df_cont.rename(columns={col: col.replace('_cont', '') for col in cols_cont},
                       inplace=True)

    sub_df_final = pd.concat([sub_df_cont, sub_df_em])

    # with plt.rc_context(bundles.iclr2023()):
    #     plt.rcParams.update(figsizes.iclr2023(nrows=2, ncols=2))

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4), sharex=True)
    axes = axes.reshape(-1)

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]
    methods = [("continuous", "--", '/'), ("EM", "o-", None)]

    T = sub_df_final["T"].unique()
    T.sort()

    for i, param in enumerate(['baseline', 'alpha', 'm', 'sigma']):
        ax = axes[i]
        # if i == 1:
        #     legend = 'auto'
        # else:
        #     legend = False
        # sns.lineplot(data=sub_df_final, x="dt", y=(param + '_err_true'),
        #              hue="T", style='estimates', palette=palette[:2],
        #              markers=['o', None], legend=legend, ax=ax)

        # hatches = ['', '//']
        # for collection, hatch in zip(ax.collections[::-1], hatches * 2):
        #     collection.set_hatch(hatch)
        # ax.set_title(dict_name_latex[param])
        # ax.set_ylabel(r'$\ell_2$ error')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_xlim(min(sub_df['dt']), max(sub_df['dt']))
        # ax.set_xlim(ax.get_xlim()[::-1])
        # ax.set_xlabel(r'$\Delta$')

        # cols = ['T', 'estimates', 'dt', 'seed'] + [f'{param}_err_true']
        # df_ax = sub_df_final[cols]

        for m, ls, hatch in methods:
            for j, t in enumerate(T):
                this_df = sub_df_final.query("T == @t and estimates == @m")
                curve = this_df.groupby("dt")[f'{param}_err_true'].quantile(
                    [0.25, 0.5, 0.75]).unstack()
                ax.loglog(
                    curve.index, curve[0.5], ls, lw=2, c=palette[j],
                    markersize=5, markevery=2
                )
                ax.fill_between(
                    curve.index, curve[0.25], curve[0.75], alpha=0.2,
                    color=palette[j], hatch=hatch, edgecolor=palette[j] if hatch else None
                )
        ax.set_xlim(1e-1, 1e-3)
        ax.set_title(dict_name_latex[param])
        if (i == 0) or (i == 2):
            ax.set_ylabel(r'$\ell_2$ error')
        if i >= 2:
            ax.set_xlabel(r'$\Delta$')

    bbox_to_anchor = (-0.2, 1.2, 1, 0.01)
    labels_m = ["EM", "Cont. EM"]
    handles_m = [plt.Line2D([], [], c="k", lw=2, marker='o', markersize=5),
                 plt.Line2D([], [], c="k", ls="--", lw=2)]
    axes[1].legend(
        handles_m,
        labels_m,
        ncol=3,
        title="Method",
        bbox_to_anchor=bbox_to_anchor,
        loc="lower left",
    )

    handles_T = [plt.Line2D([], [], c=palette[i], label=t, lw=2)
                 for i, t in enumerate(T)]
    axes[0].legend(
        handles_T,
        [r"$10^{%d}$" % np.log10(t) for t in T],
        ncol=len(T),
        title="$T$",
        bbox_to_anchor=bbox_to_anchor,
        loc="lower right",
    )
    # plt.gca().add_artist(legend_T)

    # plt.suptitle(
    #     f"Absolute estimation error for EM and continuous estimates")
    fig.tight_layout()
    plt.savefig(
        folder_name / "fig_convergence_estimates_em-true_cont_true.png",
        bbox_inches='tight')
    plt.savefig(
        folder_name / "fig_convergence_estimates_em-true_cont_true.pdf",
        bbox_inches='tight')
    plt.show()


def plot_fig1_norm2(df, err_col='err_norm2', style=None, save_fig=True):
    """

    df : pandas.DataFrame
        mandatory columns: 'err_norm2', 'T', 'dt'

    """

    palette = ['C' + str(i) for i in range(df['T'].nunique())]

    lines = sns.lineplot(
        data=df, x="dt", y=err_col, hue="T", style=style, palette=palette,
        ci='sd')

    if style is None:
        hatches = ['//']
    elif df[style].nunique() == 2:
        hatches = ['//', '']

    for collection, hatch in zip(lines.collections[::-1], hatches * 2):
        collection.set_hatch(hatch)

    plt.xscale('log')
    plt.xlim(min(df['dt']), max(df['dt']))
    plt.xlim(plt.xlim()[::-1])

    plt.yscale('log')
    plt.ylabel('norm 2 error')

    if save_fig:
        fig_name = 'estimates_norm2_err'
        plt.savefig(fig_name + ".png")
        plt.savefig(fig_name + ".pdf")

    plt.show()

    return None


def plot_fig1_paper(df):

    T = df["T"].unique()
    T.sort()
    methods = [("continuous", "--", '//'), ("EM", None, None)]

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 4)]

    with plt.rc_context(bundles.iclr2023()):

        fig = plt.figure(figsize=(bundles.iclr2023()['figure.figsize'][0],
                                  bundles.iclr2023()['figure.figsize'][1]))

        for m, ls, hatch in methods:
            for i, t in enumerate(T):
                this_df = df.query("T == @t and estimates == @m")
                curve = this_df.groupby("dt")["err_norm2"].quantile(
                    [0.25, 0.5, 0.75]).unstack()
                plt.loglog(curve.index, curve[0.5], lw=4, c=palette[i], ls=ls)
                plt.fill_between(
                    curve.index, curve[0.25], curve[0.75], alpha=0.1,
                    color=palette[i], hatch=hatch)
                plt.xlim(1e-1, 1e-3)

        # Create legend
        handles_T = [plt.Line2D([], [], c=palette[i], label=t, lw=3)
                     for i, t in enumerate(T)]
        handles_m = [plt.Line2D([], [], c="k", ls="--", lw=3),
                     plt.Line2D([], [], c="k", lw=3)]
        labels_m = ["continuous", "EM"]

        # Add legend in 2 separated boxes
        legend_T = plt.legend(
            handles_T,
            T,
            ncol=3,
            title="T",
            bbox_to_anchor=(0, 1, 1, 0.01),
            loc="lower right",
        )
        plt.legend(
            handles_m,
            labels_m,
            ncol=3,
            title="Method",
            bbox_to_anchor=(0, 1, 1, 0.01),
            loc="lower left",
        )
        plt.gca().add_artist(legend_T)
        plt.xlabel(r'$\Delta$')
        plt.ylabel(r'$\ell_2$ error')

        plt.savefig("fig1.png")
        plt.savefig("fig1.pdf")

        plt.show()


def plot_err_em_cont(df):

    T = df["T"].unique()
    T.sort()

    df['estimates'] = 'TG'
    methods = [("TG", None, None)]

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

    figsize_full = (5.5, 3.4)
    fontsize = 11
    fig = plt.figure(figsize=figsize_full)

    for m, ls, hatch in methods:
        for i, t in enumerate(T):
            this_df = df.query("T == @t and estimates == @m")
            curve = this_df.groupby("dt")["em_err_cont_norm2"].quantile(
                [0.25, 0.5, 0.75]).unstack()
            plt.loglog(curve.index, curve[0.5], lw=4, c=palette[i], ls=ls)
            plt.fill_between(
                curve.index, curve[0.25], curve[0.75], alpha=0.1,
                color=palette[i], hatch=hatch)
            plt.xlim(1e-1, 1e-3)

    # Create legend
    # handles_T = [plt.Line2D([], [], c=palette[i], label=t, lw=3)
    #              for i, t in enumerate(T)]
    # handles_m = [plt.Line2D([], [], c="k", ls="--", lw=3),
    #              plt.Line2D([], [], c="k", lw=3)]

    custom_lines_T = [plt.Line2D([], [], c=palette[i], label=t, lw=3)
                      for i, t in enumerate(T)]

    plt.legend(custom_lines_T,
               ['T={:.0e}'.format(1000), 'T={:.0e}'.format(10000)],
               fontsize=fontsize,
               bbox_to_anchor=(0.92, 1.3), ncol=2)

    plt.xlabel(r'$\Delta$')
    plt.ylabel(r'$\ell_2$ error')

    plt.savefig("fig_err_em_cont.png")
    plt.savefig("fig_err_em_cont.pdf")

    plt.show()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_folder_name(true_params, poisson_intensity, save_json=True):
    """

    """

    folder_name = SAVE_RESULTS_PATH / hash([true_params, poisson_intensity])
    if not folder_name.exists():
        folder_name.mkdir(parents=True)

    if save_json:
        experiment_param = {'poisson_intensity': poisson_intensity,
                            **true_params}
        with open(folder_name / 'experiment_param.json', 'w', encoding='utf-8') as f:
            json.dump(experiment_param, f, ensure_ascii=False,
                      indent=4, cls=NumpyEncoder)

    return folder_name


def experiment(true_params, poisson_intensity):
    """

    """
    folder_name = get_folder_name(true_params, poisson_intensity)

    df = pd.DataFrame()
    list_seed = list(range(50))
    for this_T in [1_000, 10_000]:
        new_dfs = Parallel(n_jobs=min(50, len(list_seed)), verbose=1)(
            delayed(procedure)(true_params,
                               poisson_intensity, this_T, this_seed)
            for this_seed in list_seed)
        new_dfs.append(df)
        df = pd.concat(new_dfs)
        df.to_pickle(folder_name / 'df_convergence_estimates_em.csv')

    plot_fig1(folder_name)

# %%


# define exeperiment parameters
n_drivers = 1
lower, upper = 0, 1
max_iter = 50


%matplotlib inline
%pylab inline

true_params = {
    'baseline': 3,
    'alpha': np.array([1]),
    'm': np.array([0.2]),
    'sigma': np.array([0.1])
}
poisson_intensity = 0.5
folder_name = get_folder_name(true_params, poisson_intensity)
fig_name = folder_name / "fig_convergence_estimates_em-true_cont_true.png"

plot_fig1(folder_name, L_max=10**3)

# %%

df = pd.read_pickle(folder_name / 'df_convergence_estimates_em.csv')

pre = 'em'
cols = [param + '_' + pre + '_err_' + suf
        for param in ['alpha', 'm', 'sigma']
        for suf in ['true', 'cont']]

params = ['alpha', 'm', 'sigma']
cols += [p+suf for p in params for suf in ['_cont', '_true', '_em']]

for col in cols:
    df[col] = df[col].apply(lambda x: x[0])

for param in ['baseline', 'alpha', 'm', 'sigma']:
    df[param +
        '_cont_err_true'] = np.abs(df[param + '_cont'] - df[param + '_true'])

for param in ['baseline', 'alpha', 'm', 'sigma']:
    df[param +
        '_em_err_cont'] = np.abs(df[param + '_em'] - df[param + '_cont'])

df['dt'] = 1 / df['L']


def compute_norm2_error(s, pre='em', suf='true'):

    cols = [f'{param}_{pre}_err_{suf}'
            for param in ['baseline', 'alpha', 'm', 'sigma']]

    return np.sqrt(np.array([s[this_col]**2 for this_col in cols]).sum())


df['em_err_norm2'] = df.apply(
    lambda x: compute_norm2_error(x, pre='em'), axis=1)
df['cont_err_norm2'] = df.apply(
    lambda x: compute_norm2_error(x, pre='cont'), axis=1)
df['em_err_cont_norm2'] = df.apply(
    lambda x: compute_norm2_error(x, pre='em', suf='cont'), axis=1)

plot_err_em_cont(df)

# %%

sub_df = df[df['L'] <= 1e3]


sub_df_em = sub_df[['T', 'L', 'dt', 'seed', 'em_err_norm2']]
sub_df_em['estimates'] = 'EM'
sub_df_em.rename(columns={'em_err_norm2': 'err_norm2'},
                 inplace=True)

sub_df_cont = sub_df[['T', 'L', 'dt', 'seed', 'cont_err_norm2']]
sub_df_cont['estimates'] = 'continuous'
sub_df_cont.rename(columns={'cont_err_norm2': 'err_norm2'},
                   inplace=True)

sub_df_final = pd.concat([sub_df_cont, sub_df_em])
sub_df_final.to_csv('error_discrete_EM.csv', index=False)

plot_fig1_norm2(df, err_col='em_err_norm2')
plot_fig1_norm2(sub_df_final, err_col='err_norm2', style='estimates')
plot_fig1_paper(sub_df_final)

# %%


if fig_name.exists():
    plot_fig1(folder_name, L_max=10**3)
    # img = mpimg.imread(
    #     folder_name / "fig_convergence_estimates_em-true_cont_true.png")
    # imgplot = plt.imshow(img)
    # plt.show()
else:
    experiment(true_params, poisson_intensity)


# %%


hyperparams = {
    'poisson_intensity': [0.1],  # [0.1, 0.5]
    'baseline': [1],  # [1, 2]
    'alpha': [1],
    'm': [0.2],  # [0.1, 0.5]
    'sigma': [0.1, 0.05]
}
keys, values = zip(*hyperparams.items())
permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

for params in permuts_params:
    true_params = {'baseline': params['baseline'],
                   'alpha': np.array([params['alpha']]),
                   'm': np.array([params['m']]),
                   'sigma': np.array([params['sigma']])}
    experiment(true_params, params['poisson_intensity'])

# %%


# cols_cont = [param + '_cont' for param in ['baseline', 'alpha', 'm', 'sigma']]
# # sub_df_cont = df[(df['T'] == 1000)][cols_cont + ['T', 'L', 'seed']]
# sub_df_cont = df[cols_cont + ['T', 'L', 'seed']]
# sub_df_cont['type'] = 'continuous'
# sub_df_cont.rename(columns={col: col.replace('_cont', '') for col in cols_cont},
#                    inplace=True)

# cols_em = [param + '_em' for param in ['baseline', 'alpha', 'm', 'sigma']]
# sub_df_em = df[cols_em + ['T', 'L', 'seed']]
# sub_df_em['type'] = 'discrete'
# sub_df_em.rename(columns={col: col.replace('_em', '') for col in cols_em},
#                  inplace=True)

# sub_df = pd.concat([sub_df_cont, sub_df_em])

# T = 1000

# fig, axes = plt.subplots(2, 2, figsize=(14, 8))
# axes = axes.reshape(-1)

# for i, param in enumerate(['baseline', 'alpha', 'm', 'sigma']):
#     ax = axes[i]
#     y_true = df.iloc[0][param+'_true']
#     ax.hlines(y_true, L_list[0], L_list[-1],
#               linestyle='--', label="true value")

#     sns.lineplot(data=sub_df[sub_df['T'] == T], x="L", y=param, hue="type",
#                  estimator='mean', ci='sd', ax=ax)

#     # sns.lineplot(data=sub_df, x="L", y=(param + '_em'), hue="T", ax=ax, color="green")
#     ax.legend()
#     ax.set_xscale('log')
#     ax.set_xlim(min(L_list), max(L_list))


# plt.suptitle(
#     f"Convergence of continuous and discrete estimates (T={T})")
# plt.savefig("fig_convergence_estimates.png")
# plt.show()

# %%
