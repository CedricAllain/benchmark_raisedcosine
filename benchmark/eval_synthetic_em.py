# %%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from raised_torch.model import Model
from raised_torch.kernels import compute_kernels
from raised_torch.simu_pp import simu

from raised_torch.utils.utils_plot import plot_kernels, plot_global_fig, plot_hist_params
from raised_torch.solver import initialize, compute_loss, optimizer, training_loop

from trunc_norm_kernel.optim import em_truncated_norm

baseline = 0.8
alpha = [0.8, 0.8]
m = [0.4, 0.2]
sigma = [0.2, 0.05]
isi = [1, 1.4]
# m = [0.2, 0.4]
# sigma = [0.05, 0.2]
# isi = [1.4, 1]


# alpha = [1.]
# m = [0.4]
# sigma = [0.4]

T = 10_000
L = 2_000
dt = 1 / L
p_task = 0.6
t = torch.arange(0, 1, dt)

true_params = {'baseline': baseline, 'alpha': alpha, 'sigma': sigma}
lower, upper = 0.03, 0.8

simu_kernel_name = 'gaussian'
kernel_name = 'gaussian'

if kernel_name == 'raised_cosine':
    true_params['m'] = np.array(m) - np.array(sigma)
else:
    true_params['m'] = m


true_kernels, intensity_value, driver_tt, driver, acti_tt, acti = simu(
    baseline, alpha, m, sigma, kernel_name=simu_kernel_name,
    simu_params=[T, L, p_task], lower=lower, upper=upper, isi=isi, seed=0)

# plot_kernels(true_kernels, t, title="True kernels")

# %%

# init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
#                          lower=lower, upper=upper,
#                          kernel_name=kernel_name)

init_params = [0.75, [1.1, 1.1], [0.33, 0.33], [0.18, 0.18]]
print('init params:', init_params)

baseline_init, alpha_init, m_init, sigma_init = init_params

model_raised = Model(t, baseline_init, alpha_init, m_init, sigma_init, dt,
                     kernel_name=kernel_name, lower=lower, upper=upper)
plot_kernels(model_raised.kernels, t, true_kernels=true_kernels,
             title="Initial kernels")

# %%
res_params, history_params, hist_loss = em_truncated_norm(
    acti_tt, driver_tt, lower=lower, upper=upper, T=T, sfreq=150.,
    init_params=init_params, alpha_pos=True, n_iter=1_000,
    verbose=False, disable_tqdm=False, compute_loss=True)
# %%
hist = pd.DataFrame(history_params)
print('init params:', init_params)
print('init params em:', hist.iloc[0])
plot_hist_params(hist, true_params)

plt.plot(hist_loss)
plt.show()
# %%
loss_name = 'log-likelihood'
baseline_hat, alpha_hat, m_hat, sigma_hat = res_params
model_raised = Model(t, baseline_hat, alpha_hat, m_hat, sigma_hat,
                     lower=lower, upper=upper, dt=dt,
                     kernel_name=kernel_name,
                     loss_name=loss_name)
plot_kernels(model_raised.kernels, t, true_kernels=true_kernels,
             title="Learned kernels")
# %%
