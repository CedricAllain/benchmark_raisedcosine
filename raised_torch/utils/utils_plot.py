##########################################
# Plot functions
##########################################

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns


COLOR_TRUE = 'orange'
COLOR_EST = 'blue'
COLOR_TEST = 'green'

colors = ['blue', 'orange', 'green']


def plot_kernels(kernels, t, true_kernels=None, title=None):

    if torch.is_tensor(kernels) and kernels.requires_grad:
        kernels = kernels.detach().numpy()

    if true_kernels is not None:
        if torch.is_tensor(true_kernels) and true_kernels.requires_grad:
            true_kernels = true_kernels.detach().numpy()

    for i, kernel in enumerate(kernels):
        plt.plot(t, kernel, color=colors[i], label=f"kernel {i+1}")
        if true_kernels is not None:
            plt.plot(t, true_kernels[i], linestyle='--', color=colors[i],
                     label=f"true kernel {i+1}", alpha=0.8)

    plt.title(title)
    plt.legend()
    plt.show()


def plot_global_fig(true_intensity, est_intensity, true_kernel, est_kernel,
                    pobj, test_intensity=None, pval=None,
                    loss='log-likelihood', figtitle=None):
    """

    """
    fig = plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(1, 2, figure=fig)

    # ax = fig.add_subplot(gs[0, :])
    # ax.plot(est_intensity, label="Estimated intensity", color=COLOR_EST)
    # ax.plot(true_intensity, '--', label="True intensity", color=COLOR_TRUE)
    # if test_intensity is not None:
    #     t = np.arange(len(est_intensity), len(true_intensity))
    #     ax.plot(t, test_intensity, '--',
    #             label="test intensity", color=COLOR_TEST)
    # ax.set_title("Intensity function")

    ax = fig.add_subplot(gs[0, 0])
    lns1 = ax.plot(pobj, label=f"{loss}", color=COLOR_EST)
    ax.set_ylabel(r"Train")
    if pval is not None:
        # ax2 = ax.twinx()
        lns2 = ax.plot(pval, label="test", color=COLOR_TEST)

    # added these three lines
    lns = lns1 + lns2
    labs = [ln.get_label() for ln in lns]
    ax.legend(lns, labs)

    ax = fig.add_subplot(gs[0, 1])
    for i in range(true_kernel.shape[0]):
        ax.plot(est_kernel[i], label=f'Learned kernel {i+1}', color=colors[i])
        ax.plot(true_kernel[i], '--', label=f'True kernel {i+1}',
        color=colors[i], alpha=0.8)
    ax.yaxis.tick_right()
    ax.legend()

    if figtitle is not None:
        plt.savefig(figtitle)
    plt.show()

    return fig


def plot_hist_params(hist, true_params):
    """
    Parameters
    ----------
    hist : pandas.DataFrame

    true_params : dict
    """

    colors = ['blue', 'orange', 'green']
    n_cols = 2
    fig, axes = plt.subplots(2, n_cols, figsize=(14, 8), sharex=True)
    axes = axes.reshape(-1)
    n_axes = len(axes)

    max_iter = len(hist)

    # plot baseline
    ax = axes[0]
    ax.plot(np.array(hist['baseline']), label='baseline', color=colors[0])
    ax.hlines(true_params['baseline'], 0, max_iter-1, linestyles='--',
              color=colors[0])
    # plot other parameters
    for i, param in enumerate(['alpha', 'm', 'sigma']):
        ax = axes[i+1]
        for j in range(len(true_params[param])):
            ax.plot(np.array([v[j] for v in hist[param]]),
                    label=f'{param}, kernel {j+1}', color=colors[j])
            ax.hlines(true_params[param][j], 0, max_iter-1,
                      linestyles='--', color=colors[j], alpha=0.8)
            if (i+1) in [n_axes-1, n_axes-2]:
                ax.set_xlabel('Iterations')

        ax.set_xlim(0, max_iter-1)
        ax.legend()

    plt.suptitle("Parameter history through iterations")
    plt.show()
