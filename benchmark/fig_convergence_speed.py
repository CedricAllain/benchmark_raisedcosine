"""
Plot, for multiple methods, the loss as a function of learning time.
Only works for truncated gaussian kernel and negative log-likelihood loss.
"""

# %%
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from raised_torch.simu_pp import simu
from raised_torch.model import Model
from raised_torch.solver import initialize, training_loop
from trunc_norm_kernel.optim import em_truncated_norm


# simulate data
kernel_name = 'gaussian'
loss_name = 'log-likelihood'
max_iter = 400

baseline = 1.
alpha = [1., 1.]
m = [0.4, 0.8]
sigma = [0.2, 0.05]
isi = [1, 1.4]
lower, upper = 0, 0.8

T = 100_000
L = 100
dt = 1 / L
p_task = 0.9
t = torch.arange(0, 1, dt)

kernels, intensity_value, driver_tt, driver, acti_tt, acti = simu(
    baseline, alpha, m, sigma, kernel_name=kernel_name,
    simu_params=[T, L, p_task], lower=lower, upper=upper, isi=isi, seed=0)

# initialize parameters
init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
                         lower=lower, upper=upper,
                         kernel_name=kernel_name)
baseline_init, alpha_init, m_init, sigma_init = init_params

# final dictionary, keys are method name, values are history DataFrame
dict_hist = {}
df_final = pd.DataFrame()

# learn with EM
for solver, use_dis in zip(['EM_dis', 'EM_cont'], [True, False]):
    _, hist = em_truncated_norm(
        acti_tt, driver_tt, lower=lower, upper=upper, T=T, sfreq=L,
        use_dis=use_dis, init_params=init_params, alpha_pos=True,
        n_iter=max_iter, verbose=False, disable_tqdm=False, compute_loss=True)
    # df = pd.DataFrame(hist)
    # df["solver"] = solver
    # df_final = pd.concat([df_final, df])
    dict_hist[solver] = pd.DataFrame(hist)


# %% learn with torch
# for solver in ['RMSprop', 'GD']:
#     model_raised = Model(t, baseline_init, alpha_init, m_init, sigma_init,
#                          dt=dt, kernel_name=kernel_name, loss_name=loss_name,
#                          lower=lower, upper=upper)
#     res_dict = training_loop(model_raised, driver, acti, solver=solver,
#                              step_size=1e-3, max_iter=max_iter, test=False,
#                              logging=True, device='cpu')
#     dict_hist[solver] = pd.DataFrame(res_dict['hist'])

# %% plot final figure
figsize = (14, 8)

fig = plt.figure(figsize=figsize)
for method, hist in dict_hist.items():
    plt.plot(hist['time_loop'], hist['loss'], label=method)

plt.legend()
plt.xlabel("Learning time (s.)")
plt.ylabel("Loss")
plt.title("Loss as a function of learning time")
plt.show()

# %% to compare, plot loss as a function of iteration
fig = plt.figure(figsize=figsize)
for method, hist in dict_hist.items():
    plt.semilogy(hist['loss'] - hist['loss'].min(), label=method)

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss as a function of iterations")
plt.show()

# %%
