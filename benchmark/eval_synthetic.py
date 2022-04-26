# %%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from raised_torch.model import Model
from raised_torch.simu_pp import simu
from raised_torch.utils.utils_plot import plot_kernels, plot_global_fig
from raised_torch.solver import initialize, compute_loss, optimizer, training_loop

baseline = 1.

alpha = [1., 2.]
m = [0.4, 0.8]
sigma = [0.4, 0.2]


# alpha = [1.]
# m = [0.4]
# sigma = [0.4]

T = 10_000
L = 100
dt = 1 / L
p_task = 0.3
t = torch.arange(0, 1, dt)

true_params = {'baseline': baseline, 'alpha': alpha, 'sigma': sigma}

kernel_name = 'raised_cosine'
if kernel_name == 'raised_cosine':
    true_params['m'] = np.array(m) - np.array(sigma)
else:
    true_params['m'] = m


kernels, intensity_value, driver_tt, driver, acti_tt, acti = simu(
    baseline, alpha, m, sigma,
    kernel_name=kernel_name, simu_params=[T, L, p_task])

plot_kernels(kernels, t)

init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
                         lower=0, upper=0.8,
                         kernel_name=kernel_name)

baseline_init, alpha_init, m_init, sigma_init = init_params
# %% Define model
test = 0.3


loss_name = 'log-likelihood'
solver = 'RMSprop'
step_size = 1e-3
max_iter = 8

model_raised = Model(t, baseline_init, alpha_init, m_init, sigma_init, dt,
                     kernel_name=kernel_name,
                     loss_name=loss_name)
plot_kernels(model_raised.kernels, t)

# %%
opt = optimizer(model_raised.parameters(), step_size, solver)
res_dict = training_loop(model_raised, opt, driver, acti, max_iter, test,
                         device='cpu')

# %% plot final figure
hist = pd.DataFrame(res_dict['hist'])
print(hist['alpha'])
fig = plot_global_fig(intensity_value,
                      est_intensity=res_dict['est_intensity'],
                      true_kernel=kernels,
                      est_kernel=res_dict['est_kernel'],
                      pobj=np.array(hist['loss']),
                      test_intensity=res_dict['test_intensity'],
                      pval=np.array(hist['loss_test']),
                      loss=loss_name,
                      figtitle="res_"+solver+'.pdf')

# %%


def plot_hist_params(hist, true_params):
    """
    Parameters
    ----------
    hist : pandas.DataFrame

    true_params : dict
    """

    colors = ['blue', 'orange', 'green']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.reshape(-1)

    for i, param in enumerate(['baseline', 'alpha', 'm', 'sigma']):
        ax = axes[i]
        if param == 'baseline':
            ax.plot(np.array(hist[param]),
                    label=f'{param}', color=colors[0])
            ax.hlines(true_params[param], 0, max_iter-1,
                      linestyles='--', color=colors[0])
        else:
            for j in range(len(true_params[param])):
                ax.plot(np.array([v[j] for v in hist[param]]),
                        label=f'{param}, kernel {j}', color=colors[j])
                ax.hlines(true_params[param][j], 0, max_iter-1,
                          linestyles='--', color=colors[j])

        ax.set_xlim(0, max_iter-1)
        ax.legend()

    plt.title("Parameter history through iterations")
    plt.show()


plot_hist_params(hist, true_params)
# %%
