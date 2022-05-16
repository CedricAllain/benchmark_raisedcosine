# %%
from joblib import delayed
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
import pytest
from scipy.stats import truncnorm

from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson

from raised_torch.model import Model
from raised_torch.solver import initialize, compute_loss
from raised_torch.utils.utils import check_tensor, kernel_intensity
from raised_torch.kernels import compute_kernels

from trunc_norm_kernel.model import TruncNormKernel, Intensity
from trunc_norm_kernel.metric import negative_log_likelihood

baseline = 0.01
m = [0.9]
sigma = [0.2]
lower, upper = 0, 1


def procedure(T, L, driver_tt, acti_tt, alpha=[0]):
    #
    kernel_name = "gaussian"
    dt = 1 / L
    t = np.arange(0, T, dt)
    driver = t * 0
    driver[np.round(driver_tt * L).astype(int)] += 1
    assert driver.sum() == len(driver_tt)
    driver_tt = [driver_tt]
    driver = np.array([driver])
    driver = check_tensor(driver)
    #
    kernel_support = torch.arange(lower, upper, dt)
    kernels_torch = compute_kernels(
        kernel_support, alpha, m, sigma, kernel_name, lower, upper, dt)
    for this_kernel, this_alpha in zip(kernels_torch, alpha):
        plt.plot(kernel_support, this_kernel,
                 label=f'kernel with alpha={this_alpha}')
    plt.title("True kernels")
    plt.legend()
    plt.show()
    #
    # model = Model(check_tensor(t), baseline, alpha, m, sigma, dt=dt,
    #               kernel_name=kernel_name, loss_name="log-likelihood",
    #               lower=lower, upper=upper)
    # intensity_torch = model(driver)
    kernel_dis = TruncNormKernel(
        lower, upper, m=m[0], sigma=sigma[0], sfreq=L, use_dis=True)
    kernels_torch = check_tensor([kernel_dis(kernel_support.numpy())])
    intensity_torch = kernel_intensity(baseline, driver, kernels_torch, L)
    assert len(intensity_torch) == T * L
    #
    tf = TimeFunction((t, intensity_torch.detach().numpy()), dt=dt)
    in_poi = SimuInhomogeneousPoisson(
        [tf], end_time=T, seed=0, verbose=False)
    in_poi.track_intensity(dt)
    in_poi.simulate()
    acti_tt = np.unique(np.round(in_poi.timestamps[0] / dt).astype(int) * dt)
    acti_tt.sort()
    acti = t * 0
    acti[np.round(acti_tt * L).astype(int)] += 1
    assert acti.sum() == len(acti_tt)
    print(f'{len(acti_tt)} activation simulated: {acti_tt}')
    acti = check_tensor(acti).to(torch.bool)
    #
    plt.plot(t, intensity_torch.detach().numpy())
    plt.stem(t, acti)
    plt.stem(t, driver[0], linefmt="--")
    plt.title("Intensity")
    plt.show()
    # using torch method
    log_intensity_torch = torch.log(
        intensity_torch[acti]).sum().detach().numpy()
    # using dripp method
    kernel = [TruncNormKernel(lower, upper, sfreq=L, use_dis=True)]
    intensity = Intensity(kernel=kernel, driver_tt=driver_tt, acti_tt=acti_tt)
    intensity.update(baseline, alpha, m, sigma)
    log_intensity = np.log(intensity(intensity.acti_tt,
                                     driver_delays=intensity.driver_delays)).sum()

    print("log", log_intensity_torch, log_intensity)

    model = Model(kernel_support, baseline, alpha, m, sigma, dt=dt,
              kernel_name="gaussian", loss_name='log-likelihood',
              lower=lower, upper=upper, driver=driver)
    loss_torch = compute_loss(
        "log-likelihood", intensity_torch, acti, dt, model).detach().numpy()
    loss_dripp = negative_log_likelihood(intensity, T)
    print("loss", loss_torch, loss_dripp)

    assert_almost_equal(log_intensity_torch, log_intensity, 4)
    assert_almost_equal(loss_torch, loss_dripp, 4)


# %%
T = 10
L = 100
# driver_tt = np.array([1, 1.4, 4, 4.7])
# driver_tt = np.concatenate([driver_tt + .732])
# driver_tt.sort()
driver_tt = np.array([.732])
alpha = [1]
procedure(T=4, L=100, driver_tt=driver_tt, acti_tt=None, alpha=[1])
# %%
T = 4
L = 100
dt = 1/L
m = 0.9
sigma = 0.2
baseline = 1
alpha = 1
lower, upper = 0, 1
#
driver_tt = np.array([.732])
acti_tt = np.array([1.52])
delay = acti_tt - driver_tt
# sparse driver vector
t = np.arange(0, T, dt)
driver = t * 0
driver[np.round(driver_tt * L).astype(int)] += 1
driver = np.array([driver])
driver = check_tensor(driver)
# sparse activation vector
acti = t * 0
acti[np.round(acti_tt * L).astype(int)] += 1
acti = check_tensor(acti).to(torch.bool)

# initialize kernels
kernel_cont = TruncNormKernel(
    lower, upper, m, sigma, sfreq=L, use_dis=False)
kernel_dis = TruncNormKernel(
    lower, upper, m, sigma, sfreq=L, use_dis=True)
print('kernel_cont:', kernel_cont(delay)[0])
print('kernel_dis:', kernel_dis(delay)[0])

# initialize dripp intensities
intensity_cont = Intensity(
    baseline, [alpha], kernel=kernel_cont, driver_tt=[driver_tt], acti_tt=acti_tt)
intensity_dis = Intensity(
    baseline, [alpha], kernel=kernel_dis, driver_tt=[driver_tt], acti_tt=acti_tt)
print('intensity_cont:', intensity_cont(acti_tt))
print('intensity_dis:', intensity_dis(acti_tt))

# compute true value
a = (lower - m) / sigma
b = (upper - m) / sigma
true_intensity = baseline + alpha * truncnorm.pdf(delay, a, b, m, sigma)
print(f'true_intensity = {true_intensity[0]}')

# compute loss with torch

kernel_support = torch.arange(lower, upper, dt)
kernels_torch = compute_kernels(
    kernel_support, [alpha], [m], [sigma], "gaussian", lower, upper, dt)
kernel_val = compute_kernels(check_tensor(delay), [alpha], [m],
                             [sigma], 'gaussian', lower, upper, dt)
print(f'kernels_torch = {kernel_val.numpy()}')
intensity_torch = kernel_intensity(baseline, driver, kernels_torch, L)
print(
    f'intensity_torch = {intensity_torch[acti].numpy()}')

# intensity_torch = baseline + torch.conv_transpose1d(
#         driver[None], kernels_torch[:, None]
#     )[0, 0, :-L]  # +1]
# intensity_torch = torch.concat([torch.zeros(1), intensity_torch])
# intensity_torch = intensity_torch.clip(0)
# torch.concat([torch.zeros((1,len(kernels_torch),1)), kernels_torch[:, None]])


# compute loss with torch but with dripp kernel
kernels_ = check_tensor([kernel_dis(kernel_support.numpy())])
mask_kernel = (delay < lower) | (delay > upper)
print(f'kernels_ = {kernels_[0][np.round(delay * L).astype(int)]}')
intensity_ = kernel_intensity(baseline, driver, kernels_, L)
print(f'intensity_ = {intensity_[acti].numpy()}')

# compute loss
true_loss = (T * baseline + alpha - np.log(intensity_cont(acti_tt))) / T
print('true_loss', true_loss)
loss_cont = negative_log_likelihood(intensity_cont, T)
print('loss_cont', loss_cont)
loss_dis = negative_log_likelihood(intensity_dis, T)
print('loss_dis', loss_dis)

loss_torch = (intensity_torch.sum()*dt -
              torch.log(intensity_torch[np.where(acti)[0]]).sum())/T
print('loss_torch', loss_torch.item())

model = Model(kernel_support, baseline, [alpha], [m], [sigma], dt=dt,
              kernel_name="gaussian", loss_name='log-likelihood',
              lower=lower, upper=upper, driver=driver)
print("loss_model", compute_loss("log-likelihood",
                                 intensity_torch, acti, dt, model).detach().numpy())

loss_mix = (intensity_.sum()*dt -
            torch.log(intensity_[np.where(acti)[0]]).sum())/T
print('loss_mix', loss_mix.item())

# %%
