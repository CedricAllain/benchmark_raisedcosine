from scipy.optimize import check_grad
from scipy.optimize import fmin_l_bfgs_b
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import torch

from tick.base import TimeFunction
from tick.hawkes import SimuHawkes
from tick.hawkes import HawkesKernelTimeFunc
from tick.hawkes import SimuInhomogeneousPoisson
from tick.plot import plot_point_process


def chek_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x


def raised_cosine_kernel(t, params, dt=1/1000, kernel_zero_base=False):
    """

    """
    t = chek_tensor(t)
    params = chek_tensor(params)

    _, alpha, mu, sig = params

    kernel = (1 + torch.cos((t-mu)/sig*np.pi)) / (2 * sig ** 2)
    mask_kernel = (t < (mu-sig)) | (t > (mu+sig))
    kernel[mask_kernel] = 0.
    if kernel_zero_base:
        kernel = (kernel - kernel.min())
    kernel = alpha * kernel / (kernel.sum() * dt)

    return kernel


def truncated_gaussian_kernel(t, params, lower, upper, dt=1/1000,
                              kernel_zero_base=False):
    """

    """
    t = chek_tensor(t)
    params = chek_tensor(params)

    _, alpha, mu, sig = params

    kernel = torch.exp(-(t - mu) ** 2 / sig ** 2)
    mask_kernel = (t < lower) | (t > upper)
    kernel[mask_kernel] = 0.
    if kernel_zero_base:
        kernel = (kernel - kernel.min())
    kernel = alpha * kernel / (kernel.sum() * dt)

    return kernel

# %%


def simu(true_params, simu_params=[50, 1000, 0.5], seed=None,
         plot_intensity=True,):
    """

    Parameters
    ----------


    Returns
    -------
    intensity_csc

    z : array-like
        sparse vector where 1 indicates an intensity activation
    """

    mu_0, alpha_true, mu_true, sig_true = true_params
    T, L, p_task = simu_params
    dt = 1 / L

    # simulate data
    t_value = np.linspace(0, 1, L + 1)[:-1]
    y_value = np.array(raised_cosine_kernel(t_value, true_params, dt))

    # generate driver timestamps
    isi = 0.7
    t_k = np.arange(start=0, stop=T - 2 * isi, step=isi)
    # sample timestamps
    rng = np.random.RandomState(seed=seed)
    t_k = rng.choice(t_k, size=int(p_task * len(t_k)),
                     replace=False).astype(float)
    t_k = (t_k / dt).astype(int) * dt
    # create sparse vector
    t = np.arange(0, T + 1e-10, dt)
    driver_tt = t * 0
    driver_tt[(t_k * L).astype(int)] += 1
    intensity_csc = mu_0 + np.convolve(driver_tt, y_value)[:-L+1]

    #
    tf = TimeFunction((t, intensity_csc), dt=dt)
    # We define a 1 dimensional inhomogeneous Poisson process with the
    # intensity function seen above
    in_poi = SimuInhomogeneousPoisson(
        [tf], end_time=T, seed=seed, verbose=False)
    # We activate intensity tracking and launch simulation
    in_poi.track_intensity(dt)
    in_poi.simulate()

    # We plot the resulting inhomogeneous Poisson process with its
    # intensity and its ticks over time
    if plot_intensity:
        plot_point_process(in_poi)

    t_k = (in_poi.timestamps[0] / dt).astype(int) * dt
    acti_tt = t * 0
    acti_tt[(t_k * L).astype(int)] += 1

    return y_value, intensity_csc, driver_tt, acti_tt, in_poi


def compute_loss(loss, intensity, acti_t, dt):
    """
    """
    T = len(intensity) * dt
    if loss == 'log-likelihood':
        # negative log-likelihood
        return (intensity.sum() * dt - torch.log(intensity[acti_t]).sum()) / T
    elif loss == 'MSE':
        return ((intensity ** 2).sum() * dt - 2 * (intensity[acti_t]).sum()) / T
    else:
        raise ValueError(
            f"loss must be 'MLE' or 'log-likelihood', got '{loss}'"
        )


def run_gd(driver_tt, acti_tt, L=1000, init_params=None,
           loss='log-likelihood', kernel_zero_base=False, max_iter=100,
           step_size=1e-5, test=0.3):
    """

    Parameters
    ----------
    true_params : list
        [mu_0, alpha_true, mu_true, sig_true]



    """
    dt = 1/L
    # intialize parameters
    P0 = torch.tensor(init_params, requires_grad=True)
    mu0 = P0[0]

    # define kernel support
    t = torch.arange(0, 1, dt)

    pobj = []
    pval = []
    driver_torch_temp = torch.tensor(driver_tt).float()
    acti_torch_temp = torch.tensor(acti_tt).float()

    if test:
        n_test = np.int(np.round(test * len(driver_torch_temp)))
        driver_torch = driver_torch_temp[:-n_test]
        driver_torch_test = driver_torch_temp[-n_test:]
        driver_t_test = driver_torch_test.to(torch.bool)

        acti_torch = acti_torch_temp[:-n_test]
        acti_torch_test = acti_torch_temp[-n_test:]
        acti_t_test = acti_torch_test.to(torch.bool)
    else:
        driver_torch = driver_torch_temp
        acti_torch = acti_torch_temp

    driver_t = driver_torch.to(torch.bool)
    acti_t = acti_torch.to(torch.bool)

    start = time.time()
    for i in range(max_iter):
        print(f"Fitting model... {i/max_iter:6.1%}\r", end='', flush=True)
        P0.grad = None
        # kernel = torch.exp(-(t - mu) ** 2 / sig ** 2)
        kernel = raised_cosine_kernel(t, P0, dt, kernel_zero_base)
        # torch.exp(mu0)
        intensity = mu0 + torch.conv_transpose1d(driver_torch[None, None],
                                                 kernel[None, None]
                                                 #    padding=(L-1,)
                                                 )[0, 0, :-L+1]

        v_loss = compute_loss(loss, intensity, acti_t, dt)
        v_loss.backward()
        P0.data -= step_size * P0.grad
        P0.data[1] = max(0, P0.data[1])  # alpha
        P0.data[3] = max(0, P0.data[3])  # sigma

        pobj.append(v_loss.item())
        if test:
            intensity_test = mu0 + torch.conv_transpose1d(driver_torch_test[None, None],
                                                          kernel[None, None]
                                                          #    padding=(L-1,)
                                                          )[0, 0, :-L+1]
            pval.append(compute_loss(
                loss, intensity_test, acti_t_test, dt).item())

    print(f"Fitting model... done ({np.round(time.time()-start)} s.) ")
    print(f"Estimated parameters: {np.array(P0.data)}")

    res_dict = {'est_intensity': np.array(intensity.detach()),
                'est_kernel': np.array(kernel.detach()),
                'pobj': pobj,
                'est_params': P0.data}
    if test:
        res_dict.update(test_intensity=np.array(intensity_test.detach()),
                        pval=pval, n_test=n_test)

    return res_dict


COLOR_TRUE = 'orange'
COLOR_EST = 'blue'
COLOR_TEST = 'green'


def plot_global_fig(true_intensity, est_intensity, true_kernel, est_kernel,
                    pobj, test_intensity=None, pval=None, loss='log-likelihood'):
    """

    """
    fig = plt.figure()
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

if __name__ == '__main__':

    # true parameters
    mu_0 = 0.7
    alpha_true = 0.7
    mu_true = 0.3
    sig_true = 0.2
    true_params = np.array([mu_0, alpha_true, mu_true, sig_true])
    # simulation parameters
    T = 100
    L = 1000
    p_task = 0.3
    # simulate data
    seed = 42
    true_kernel, true_intensity, driver_tt, acti_tt, in_poi = simu(
        true_params, simu_params=[T, L, p_task], seed=seed, plot_intensity=True)
    # int to have a train/test split, otherwise set at 0 or False
    test = 0.3  
    # initialize parameters
    rng = np.random.RandomState(seed=seed)
    p = 0.2  # init parameters are +- p% around true parameters
    init_params = rng.uniform(low=true_params*(1-p), high=true_params*(1+p))
    init_params = init_params.clip(1e-5)
    print(f"True parameters: {true_params}")
    print(f"Initial parameters: {init_params}")
    # run gradient descent
    loss = 'log-likelihood'  # 'log-likelihood' | 'MSE'
    res_dict = run_gd(driver_tt, acti_tt, L=L, init_params=init_params,
                      loss=loss, kernel_zero_base=False, max_iter=100,
                      step_size=1e-2)
    # plot final figure
    fig = plot_global_fig(true_intensity, est_intensity=res_dict['est_intensity'],
                          true_kernel=true_kernel, est_kernel=res_dict['est_kernel'],
                          pobj=res_dict['pobj'], test_intensity=res_dict['test_intensity'],
                          pval=res_dict['pval'], loss=loss)

    driver_torch_temp = torch.tensor(driver_tt).double()
    acti_torch_temp = torch.tensor(acti_tt).double()

    if test:
        n_test = np.int(np.round(test * len(driver_torch_temp)))
        driver_torch = driver_torch_temp[:-n_test]
        driver_torch_test = driver_torch_temp[-n_test:]
        driver_t_test = driver_torch_test.to(torch.bool)

        acti_torch = acti_torch_temp[:-n_test]
        acti_torch_test = acti_torch_temp[-n_test:]
        acti_t_test = acti_torch_test.to(torch.bool)
    else:
        driver_torch = driver_torch_temp
        acti_torch = acti_torch_temp

    # driver_t = driver_torch.to(torch.bool)
    acti_t = acti_torch.to(torch.bool)


    rng = np.random.RandomState(seed=seed)
    p = .2  # init parameters are +- p% around true parameters
    init_params = rng.uniform(low=true_params*(1-p), high=true_params*(1+p))
    print(init_params)

    def nll(params):

        P0 = torch.tensor(params.copy(), requires_grad=True, dtype=torch.float64)
        mu0 = P0[0]

        # define kernel support
        dt = 1/L
        t = torch.arange(0, 1, dt, dtype=torch.float64)

        P0.grad = None
        kernel = raised_cosine_kernel(t, P0, dt, kernel_zero_base=False)
        # torch.exp(mu0)
        intensity = mu0 + torch.conv_transpose1d(driver_torch[None, None],
                                                 kernel[None, None]
                                                 )[0, 0, :-L+1]
        v_loss = compute_loss(loss, intensity, acti_t, dt)
        v_loss.backward()
        assert v_loss.dtype == torch.double

        return v_loss.item(), P0.grad.numpy() * 1e-2

    # currently fmin_l_bfgs_b is not taken into account in the global figure
    fmin_l_bfgs_b(nll, init_params, bounds=[(0, None)]*4, iprint=99)
