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
from raised_torch.solver import initialize, training_loop, compute_loss
from raised_torch.kernels import compute_kernels
from raised_torch.utils.utils import check_tensor, kernel_intensity

from trunc_norm_kernel.optim import em_truncated_norm
from trunc_norm_kernel.model import TruncNormKernel, Intensity
from trunc_norm_kernel.metric import negative_log_likelihood

# simulate data
kernel_name = 'gaussian'
loss_name = 'log-likelihood'
max_iter = 100

baseline = 1.
alpha = [1., 1.]
m = [0.4, 0.8]
sigma = [0.2, 0.05]
isi = [1, 1.4]
lower, upper = 0, 0.8

T = 10_000
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
model = Model(t, baseline_init, alpha_init, m_init, sigma_init,
              dt=dt, kernel_name=kernel_name, loss_name=loss_name,
              lower=lower, upper=upper, driver=driver)
for solver in ['RMSprop']:  # , 'GD']:
    res_dict = training_loop(model, driver, acti, solver=solver,
                             step_size=1e-3, max_iter=max_iter, test=False,
                             logging=True, device='cpu')
    dict_hist[solver] = pd.DataFrame(res_dict['hist'])

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
for method, hist in dict_hist.items():
    print(hist['loss'][0])

# %%
kernel = [TruncNormKernel(lower, upper, sfreq=L, use_dis=True)
          for _ in range(len(alpha))]
intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)
intensity.update(baseline_init, alpha_init, m_init, sigma_init)
nll_dripp = negative_log_likelihood(intensity, T)
print(f"nll_dripp = {nll_dripp}")

kernel_support = torch.arange(lower, upper, dt)
kernels_torch = compute_kernels(
    kernel_support, alpha_init, m_init, sigma_init, "gaussian", lower, upper, dt)
intensity_torch = kernel_intensity(baseline_init, driver, kernels_torch, L)
nll_torch = compute_loss("log-likelihood", intensity_torch,
                         acti, dt, model).detach().numpy()
print(f"nll_torch = {nll_torch}")
# %%
