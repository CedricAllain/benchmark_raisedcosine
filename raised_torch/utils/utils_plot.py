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


def plot_kernels(kernels, t):

    if torch.is_tensor(kernels) and kernels.requires_grad:
        kernels = kernels.detach().numpy()

    for i, kernel in enumerate(kernels):
        plt.plot(t, kernel, label=f"kernel {i+1}")
    plt.legend()


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
        ax.plot(est_kernel[i], label=f'Learned kernel {i}', color=COLOR_EST)
        ax.plot(true_kernel[i], '--',
                label=f'True kernel {i}', color=COLOR_TRUE)
    ax.yaxis.tick_right()
    ax.legend()

    if figtitle is not None:
        plt.savefig(figtitle)
    plt.show()

    return fig


def plot_hist_params(hist_params):
    """

    Parameters
    ----------
    hist_params : dict of dict
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for i, param in enumerate(hist_params.keys()):
        axes = 