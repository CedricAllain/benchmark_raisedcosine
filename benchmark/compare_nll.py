# %%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from joblib import Memory, Parallel, delayed

from raised_torch.simu_pp import simu
from raised_torch.model import Model
from raised_torch.solver import initialize, compute_loss
from raised_torch.utils.utils import check_tensor

from trunc_norm_kernel.model import TruncNormKernel, Intensity
from trunc_norm_kernel.metric import negative_log_likelihood

N_JOBS = 4

# simulate data
kernel_name = 'gaussian'
loss_name = 'log-likelihood'
max_iter = 400
device = 'cpu'

n_drivers = 2
baseline = 1.
alpha = [1., 1.]
m = [0.4, 0.8]
sigma = [0.2, 0.05]
isi = [1, 1.4]
lower, upper = 0, 0.8

dict_loss = []
T = 10_000
L = 100


def procedure(T, L, verbose=False):
    dt = 1 / L
    p_task = 0.6
    t = torch.arange(0, 1, dt)

    _, _, driver_tt, driver, acti_tt, acti = simu(
        baseline, alpha, m, sigma, kernel_name=kernel_name,
        simu_params=[T, L, p_task], lower=lower, upper=upper, isi=isi, seed=0)

    # initialize parameters
    init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
                             lower=lower, upper=upper,
                             kernel_name=kernel_name)
    baseline_init, alpha_init, m_init, sigma_init = init_params

    # close form of the integral
    integ = baseline_init * T
    for this_alpha, this_driver_tt in zip(alpha_init, driver_tt):
        integ += this_alpha * len(this_driver_tt)

    # %% compute nll with DriPP without discretization
    kernel = [TruncNormKernel(lower, upper, use_dis=False)
              for _ in range(n_drivers)]
    intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)
    intensity.update(baseline_init, alpha_init, m_init, sigma_init)
    nll_dripp_cont = negative_log_likelihood(intensity, T)
    log_intensity = np.log(intensity(intensity.acti_tt,
                                     driver_delays=intensity.driver_delays)).sum()

    # compute nll with DriPP with discretization
    kernel = [TruncNormKernel(lower, upper, sfreq=L, use_dis=True)
              for _ in range(n_drivers)]
    intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)
    intensity.update(baseline_init, alpha_init, m_init, sigma_init)
    nll_dripp_dis = negative_log_likelihood(intensity, T)

    # %% compute nll with torch
    driver = check_tensor(driver).to(device)
    acti = check_tensor(acti).to(device).to(torch.bool)

    model = Model(t, baseline_init, alpha_init, m_init, sigma_init, dt=dt,
                  kernel_name=kernel_name, loss_name=loss_name,
                  lower=lower, upper=upper)
    model = model.to(device)
    intensity_torch = model(driver)
    nll_torch = compute_loss(
        model.loss_name, intensity_torch, acti, model.dt).item()
    # compute integral estimation
    if verbose:
        print(f'L={L}, T={T}')
        print("integ =", integ)
        print("integ_est =", intensity_torch.sum()*dt)
        print("log intensity at acti =", log_intensity)
        print("torch log intensity at acti =",
              torch.log(intensity_torch[acti]).sum())

    dict_res = dict(dripp_cont=nll_dripp_cont,
                    dripp_dis=nll_dripp_dis,
                    torch=nll_torch,
                    T=T, L=L)
    return dict_res


# %% Compute loss difference while varying L (4.4 min)
L_list = [100, 200, 500, 1_000]
df_loss_L = Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(procedure)(T, this_L) for this_L in L_list)
df_loss_L = pd.DataFrame(df_loss_L)

for method in ['dripp_cont', 'dripp_dis', 'torch']:
    plt.plot(df_loss_L['L'], df_loss_L[method], label=method)
plt.xlim(min(L_list), max(L_list))
plt.xlabel('L')
plt.xticks(L_list)
plt.title(f'{loss_name} obtained with 3 methods')
plt.legend()
plt.show()

# %% Compute loss difference while varying T (2.6 s)
T_list = [100, 500, 1_000, 5_000, 10_000]
df_loss_T = Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(procedure)(this_T, L) for this_T in T_list)
df_loss_T = pd.DataFrame(df_loss_T)

for method in ['dripp_cont', 'dripp_dis', 'torch']:
    plt.plot(df_loss_T['T'], df_loss_T[method], label=method)
plt.xlim(min(T_list), max(T_list))
plt.xlabel('T')
plt.xscale('log')
plt.xticks(T_list)
plt.title(f'{loss_name} from dripp continuous')
plt.legend()
plt.show()
# %%
