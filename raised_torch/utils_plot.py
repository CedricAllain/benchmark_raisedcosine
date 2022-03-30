##########################################
# Plot functions
##########################################

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


COLOR_TRUE = 'orange'
COLOR_EST = 'blue'
COLOR_TEST = 'green'


def check_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x


def plot_global_fig(true_intensity, est_intensity, true_kernel, est_kernel,
                    pobj, test_intensity=None, pval=None, loss='log-likelihood'):
    """

    """
    fig = plt.figure(figsize=(14,8))
    gs = plt.GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(est_intensity, label="Estimated intensity", color=COLOR_EST)
    ax.plot(true_intensity, '--', label="True intensity", color=COLOR_TRUE)
    if test_intensity is not None:
        t = np.arange(len(est_intensity), len(true_intensity))
        ax.plot(t, test_intensity, '--',
                label="test intensity", color=COLOR_TEST)
    ax.set_title("Intensity function")

    ax = fig.add_subplot(gs[1, 0])
    lns1 = ax.plot(pobj, label=f"{loss}", color=COLOR_EST)
    ax.set_ylabel(r"Train")
    if pval is not None:
        # ax2 = ax.twinx()
        lns2 = ax.plot(pval, label="test", color=COLOR_TEST)

    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(est_kernel, label='Learned kernel', color=COLOR_EST)
    ax.plot(true_kernel, '--', label='True kernel', color=COLOR_TRUE)
    ax.yaxis.tick_right()
    ax.legend()

    plt.show()

    return fig