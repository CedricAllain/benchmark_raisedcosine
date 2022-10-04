# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# from tueplots.bundles.iclr2023()
figsize_full = (5.5, 3.4)
fontsize = 11

df = pd.read_csv('error_discrete_EM.csv')
# %%


def plot_fig1_paper(kernel='TG'):
    """
    kernel : str
        'TG' | 'EM'
    """

    df = pd.read_csv(f'error_discrete_{kernel}.csv')

    T = df["T"].unique()
    T.sort()

    if kernel == 'EM':
        methods = [("continuous", "--", '//'), ("EM", None, None)]
    elif kernel == 'TG':
        df['estimates'] = 'TG'
        methods = [("TG", None, None)]

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

    fig = plt.figure(figsize=figsize_full)

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

    # Add legend in 2 separated boxes
    if kernel == 'EM':
        labels_m = ["continuous", "EM"]
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

    elif kernel == 'TG':
        custom_lines_T = [Line2D([0], [0], color=palette[0], lw=3),
                          Line2D([0], [0], color=palette[1], lw=3),
                          Line2D([0], [0], color=palette[2], lw=3),
                          Line2D([0], [0], color=palette[3], lw=3)]

        plt.legend(custom_lines_T, ['T={:.0e}'.format(1000), 'T={:.0e}'.format(10000),
                                    'T={:.0e}'.format(100000), 'T={:.0e}'.format(1000000)], fontsize=fontsize,
                   bbox_to_anchor=(0.92, 1.3), ncol=2)

    plt.xlabel(r'$\Delta$')
    plt.ylabel(r'$\ell_2$ error')

    plt.savefig("fig1.png")
    plt.savefig("fig1.pdf")

    plt.show()


plot_fig1_paper(kernel='TG')
plot_fig1_paper(kernel='EM')
# %%
