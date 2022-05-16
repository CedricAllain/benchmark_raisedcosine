# %%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from joblib import Memory, Parallel, delayed

from raised_torch.simu_pp import simu
from raised_torch.model import Model
from raised_torch.solver import initialize, compute_loss
from raised_torch.utils.utils import check_tensor, kernel_intensity

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
    # m = [0.5]
    m = [0.9]
    sigma = [0.2]
    isi = [1]
elif n_drivers == 2:
    baseline = 1
    alpha = [1., 1]
    m = [0.5, 0.9]
    sigma = [0.1, 0.2]
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
    kernel_cont = [TruncNormKernel(lower, upper, use_dis=False)
                   for _ in range(n_drivers)]
    intensity_cont = Intensity(
        kernel=kernel_cont, driver_tt=driver_tt, acti_tt=acti_tt)
    intensity_cont.update(baseline, alpha, m, sigma)
    nll_dripp_cont = negative_log_likelihood(intensity_cont, T)
    log_intensity_cont = np.log(intensity_cont(intensity_cont.acti_tt,
                                               driver_delays=intensity_cont.driver_delays)).sum()

    # compute nll with DriPP with discretization
    kernel_dis = [TruncNormKernel(lower, upper, sfreq=L, use_dis=True)
                  for _ in range(n_drivers)]
    intensity_dis = Intensity(
        kernel=kernel_dis, driver_tt=driver_tt, acti_tt=acti_tt)
    intensity_dis.update(baseline, alpha, m, sigma)
    nll_dripp_dis = negative_log_likelihood(intensity_dis, T)
    log_intensity_dis = np.log(intensity_dis(intensity_dis.acti_tt,
                                             driver_delays=intensity_dis.driver_delays)).sum()

    # compute nll with torch
    driver = check_tensor(driver).to(device)
    acti = check_tensor(acti).to(device).to(torch.bool)

    model = Model(kernel_support, baseline, alpha, m, sigma, dt=dt,
                  kernel_name=kernel_name, loss_name=loss_name,
                  lower=lower, upper=upper)
    # model = model.to(device)
    intensity_torch = model(driver)
    nll_torch = compute_loss(
        model.loss_name, intensity_torch, acti, model.dt).item()
    if verbose:
        t_max = 5
        plt.plot(t[:t_max*L], intensity_torch.detach().numpy()[:t_max*L])
        plt.stem(t[:t_max*L], acti[:t_max*L])
        plt.title("Intensity")
        plt.show()

    # compute nll with torch based on dripp kernel
    kernels_ = check_tensor([this_kernel(kernel_support.numpy())
                             for this_kernel in kernel_dis])
    intensity_mix = kernel_intensity(baseline, driver, kernels_, L)
    nll_mix = compute_loss(
        model.loss_name, intensity_mix, acti, model.dt).item()

    # compute integral estimation
    if verbose:
        print(f'L={L}, T={T}, p_task={p_task}')
        print("integ =", integ)
        print("integ_est =", (intensity_torch.sum()*dt).item())
        print("integ_mix =", (intensity_mix.sum()*dt).item())
        print("log intensity at acti cont =", log_intensity_cont)
        print("log intensity at acti dis =", log_intensity_dis)
        print("torch log intensity at acti =",
              torch.log(intensity_torch[acti]).sum().item())
        print("mix log intensity at acti =",
              torch.log(intensity_mix[acti]).sum().item())
        print(f'nll_dripp_cont = {nll_dripp_cont}')
        print(f'nll_dripp_dis = {nll_dripp_dis}')
        print(f'nll_torch = {nll_torch}')
        print(f'nll_mix = {nll_mix}')

    dict_res = dict(dripp_cont=nll_dripp_cont,
                    dripp_dis=nll_dripp_dis,
                    torch=nll_torch,
                    mix=nll_mix,
                    T=T, L=L, p_task=p_task)
    return dict_res


def plot_losses(df_loss, methods, param_list, param, diff=True, xlogscale=False):
    for method in methods:
        if diff:
            yy = df_loss[method] - df_loss['dripp_cont']
        else:
            yy = df_loss[method]
        plt.plot(df_loss[param], yy, label=method)
    plt.xlim(min(param_list), max(param_list))
    plt.xlabel(param)
    if xlogscale:
        plt.xscale('log')
    plt.xticks(param_list)
    if diff:
        plt.title(f'diff of loss from continuous evaluation')
    else:
        plt.title(f'{loss_name} obtained with 3 methods')
    plt.legend()
    plt.show()


_ = procedure(T=10_000, L=200, p_task=0.6, verbose=True)
# methods = ['dripp_cont', 'dripp_dis', 'torch', 'mix']
methods = ['torch', 'mix']
# %% Compute loss difference while varying p_task
p_list = np.arange(0, 1.1, step=0.1)
df_loss_p = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(T=10_000, L=200, p_task=this_p) for this_p in p_list)
df_loss_p = pd.DataFrame(df_loss_p)

plot_losses(df_loss_p, methods, p_list, "p_task", diff=True)

# %% Compute loss difference while varying L (4.4 min)
L_list = [100, 200, 500, 1_000]
df_loss_L = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(T, this_L) for this_L in L_list)
df_loss_L = pd.DataFrame(df_loss_L)

plot_losses(df_loss_L, methods, L_list, "L", diff=True, xlogscale=False)

# %% Compute loss difference while varying T (2.6 s)
T_list = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
df_loss_T = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(T=this_T, L=200, p_task=0.6) for this_T in T_list)
df_loss_T = pd.DataFrame(df_loss_T)

plot_losses(df_loss_T, methods, T_list, "T", diff=True, xlogscale=True)

# %%
