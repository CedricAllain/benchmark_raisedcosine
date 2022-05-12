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
if n_drivers == 1:
    n_drivers = 1
    baseline = 2.
    alpha = [1.]
    # alpha = [0]
    m = [0.4]
    sigma = [0.2]
    isi = [1]
elif n_drivers == 2:
    baseline = 1
    alpha = [1., 1]
    m = [0.4, 0.6]
    sigma = [0.1, 0.05]
    isi = [1, 1]

lower, upper = 0, 1

dict_loss = []
T = 10_000
L = 100


def procedure(T, L, p_task=0.6, verbose=False):
    dt = 1 / L
    kernel_support = torch.arange(0, 1, dt)
    t = np.arange(0, T, dt)

    _, _, driver_tt, driver, acti_tt, acti = simu(
        baseline, alpha, m, sigma, kernel_name=kernel_name,
        simu_params=[T, L, p_task], lower=lower, upper=upper, isi=isi, seed=0)
    if verbose:
        print(f'#events on driver 1: {len(driver_tt[0])}')

    # test
    tt = np.where(acti == 1)[0] / L
    idx = np.where(np.abs(acti_tt - tt) > 0.01)[0]
    assert len(idx) == 0

    # close form of the integral
    integ = baseline * T
    for this_alpha, this_driver_tt in zip(alpha, driver_tt):
        integ += this_alpha * len(this_driver_tt)

    # %% compute nll with DriPP without discretization
    kernel = [TruncNormKernel(lower, upper, use_dis=False)
              for _ in range(n_drivers)]
    intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)
    intensity.update(baseline, alpha, m, sigma)
    nll_dripp_cont = negative_log_likelihood(intensity, T)
    log_intensity_cont = np.log(intensity(intensity.acti_tt,
                                     driver_delays=intensity.driver_delays)).sum()

    # compute nll with DriPP with discretization
    kernel = [TruncNormKernel(lower, upper, sfreq=L, use_dis=True)
              for _ in range(n_drivers)]
    intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)
    intensity.update(baseline, alpha, m, sigma)
    nll_dripp_dis = negative_log_likelihood(intensity, T)
    log_intensity_dis = np.log(intensity(intensity.acti_tt,
                                     driver_delays=intensity.driver_delays)).sum()

    # %% compute nll with torch
    driver = check_tensor(driver).to(device)
    acti = check_tensor(acti).to(device).to(torch.bool)

    model = Model(kernel_support, baseline, alpha, m, sigma, dt=dt,
                  kernel_name=kernel_name, loss_name=loss_name,
                  lower=lower, upper=upper)
    # model = model.to(device)
    intensity_torch = model(driver)
    if verbose:
        t_max = 10
        plt.plot(t[:t_max*L], intensity_torch.detach().numpy()[:t_max*L])
        plt.stem(t[:t_max*L], acti[:t_max*L])
        plt.title("Intensity")
        plt.show()
    nll_torch = compute_loss(
        model.loss_name, intensity_torch, acti, model.dt).item()
    # compute integral estimation
    if verbose:
        print(f'L={L}, T={T}, p_task={p_task}')
        print("integ =", integ)
        print("integ_est =", intensity_torch.sum()*dt)
        print("log intensity at acti cont =", log_intensity_cont)
        print("log intensity at acti dis =", log_intensity_dis)
        print("torch log intensity at acti =",
              torch.log(intensity_torch[acti]).sum())
        print(f'nll_dripp_dis={nll_dripp_dis}')
        print(f'nll_torch={nll_torch}')

    dict_res = dict(dripp_cont=nll_dripp_cont,
                    dripp_dis=nll_dripp_dis,
                    torch=nll_torch,
                    T=T, L=L, p_task=p_task)
    return dict_res


_ = procedure(T=10_000, L=100, p_task=0.6, verbose=True)

# %% Compute loss difference while varying p_task
p_list = np.arange(0, 1.1, step=0.1)
df_loss_p = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(T=10_000, L=200, p_task=this_p) for this_p in p_list)
df_loss_p = pd.DataFrame(df_loss_p)

for method in ['dripp_cont', 'dripp_dis', 'torch']:
    plt.plot(df_loss_p['p_task'], df_loss_p[method]-df_loss_p['dripp_cont'],
             label=method)
plt.xlim(min(p_list), max(p_list))
plt.xlabel('p_taks')
plt.xticks(p_list)
plt.title(f'{loss_name} obtained with 3 methods')
plt.legend()
plt.show()

# %% Compute loss difference while varying L (4.4 min)
L_list = [100, 200, 500, 1_000]
df_loss_L = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(T, this_L) for this_L in L_list)
df_loss_L = pd.DataFrame(df_loss_L)

for method in ['dripp_cont', 'dripp_dis', 'torch']:
    plt.plot(df_loss_L['L'], df_loss_L[method]-df_loss_L['dripp_cont'], label=method)
plt.xlim(min(L_list), max(L_list))
plt.xlabel('L')
plt.xticks(L_list)
plt.title(f'{loss_name} obtained with 3 methods')
plt.legend()
plt.show()

# %% Compute loss difference while varying T (2.6 s)
T_list = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
df_loss_T = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(this_T, L, p_task=0.6) for this_T in T_list)
df_loss_T = pd.DataFrame(df_loss_T)

for method in ['dripp_cont', 'dripp_dis', 'torch']:
    plt.plot(df_loss_T['T'], df_loss_T[method]-df_loss_T['dripp_cont'], label=method)
plt.xlim(min(T_list), max(T_list))
plt.xlabel('T')
plt.xscale('log')
plt.xticks(T_list)
plt.title(f'{loss_name} from dripp continuous')
plt.legend()
plt.show()
# %%
