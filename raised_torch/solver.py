##########################################
# Useful functions for optimization
##########################################

import numpy as np
import torch
import torch.optim as optim
import time

from scipy.sparse import csr_matrix
from scipy.sparse import find

from .utils.utils import check_tensor, check_driver_tt
EPS = np.finfo(float).eps


def get_driver_delays(driver_tt, t, lower=0, upper=1):
    """For each driver, compute the sparse delay matrix between the time(s) t
    and the driver timestamps.

    Parameters
    ----------
    driver_tt : 2d array-like

    intensity : instance of model.Intensity
    t : int | float | array-like
    Returns
    -------
    list of scipy.sparse.csr_matrix
    """

    t = np.atleast_1d(t)

    delays = []
    for this_driver_tt in driver_tt:
        # lower, upper = intensity.kernel[p].lower, intensity.kernel[p].upper
        # Construct a sparse matrix
        this_driver_delays = []
        indices = []
        indptr = [0]
        n_col = 1  # number of columns of the full sparse matrix
        for this_t in t:
            this_t_delays = this_t - this_driver_tt[(
                this_driver_tt >= this_t - upper) & (this_driver_tt <= this_t - lower)]
            n_delays = len(this_t_delays)
            n_col = max(n_col, n_delays)
            if n_delays > 0:
                this_driver_delays.extend(this_t_delays)
                indices.extend(list(range(n_delays)))

            indptr.append(indptr[-1] + n_delays)

        n_delays = len(this_driver_delays)
        if indptr[-1] != n_delays:
            indptr.append(n_delays)  # add termination

        # create sparse matrix
        M = csr_matrix((np.array(this_driver_delays), np.array(
            indices), np.array(indptr)), shape=(len(t), n_col))
        delays.append(M)

    return delays


def compute_lebesgue_support(all_tt, lower, upper):
    """Compute the Lebesgue measure of the union of the kernels supports
    following a set of timestamps.
    Compute lebesgue_measure(Union{[tt + lower, tt + upper] for tt in all_tt})
    Parameters
    ----------
    all_tt : array-like
        The set of all timestamps that induce a kernel support.
    lower, upper : float
        Lower and upper bounds of the truncated gaussian kernel.
    Returns
    -------
    float
        The Lesbegue measure of the supports union.
    """
    s = 0
    temp = (all_tt[0] + lower, all_tt[0] + upper)
    for i in range(all_tt.size - 1):
        if all_tt[i+1] + lower > temp[1]:
            s += temp[1] - temp[0]
            temp = (all_tt[i+1] + lower, all_tt[i+1] + upper)
        else:
            temp = (temp[0], all_tt[i+1] + upper)

    s += temp[1] - temp[0]
    return s


def initialize_baseline(driver_delays, driver_tt, acti_tt, T, lower, upper):
    """ Initialize the baseline parameter with a smart strategy.
    The initial value correspond of the average number of activations that lend
    outside any kernel support.
    Parameters
    ----------
    intensity : instance of model.Intensity
        The intensity object that contains the different drivers.
    T : int | float | None
        Duration of the process. If None, is set to the maximum the intensity
        activation timestamps plus a margin equal to the upper truncation
        value. Defaults to None.
    Returns
    -------
    float
        The initial value of the the baseline parameter with a smart strategy.
    """
    # compute the number of activation that lend in at least one kernel's
    # support
    acti_in_support = []
    for delays in driver_delays:
        # get the colons (i.e., the activation tt) for wich there is at least
        # one "good" delay)
        acti_in_support.extend(find(delays)[0])

    # compute the Lebesgue measure of all kernels' supports
    all_tt = np.sort(np.hstack(driver_tt))
    s = compute_lebesgue_support(all_tt, lower, upper)
    if T is None:
        T = acti_tt.max() + upper
    baseline_init = (len(acti_tt) -
                     len(set(acti_in_support))) / (T - s)
    return baseline_init


def initialize(driver_tt, acti_tt, T, initializer='smart_start', lower=0,
               upper=1, kernel_name='raised_cosine'):
    """

    """

    n_drivers = len(driver_tt)
    driver_delays = get_driver_delays(driver_tt, acti_tt, lower, upper)

    # default values
    if kernel_name == 'raised_cosine':
        default_m = upper / 2
        default_sigma = upper / 2
    elif kernel_name == 'gaussian':
        default_m = (upper - lower) / 2
        default_sigma = 0.95 * (upper - lower) / 4

    if acti_tt.size == 0:  # no activation, default values
        baseline_init = 0
        alpha_init = np.full(n_drivers, fill_value=0)
        m_init = np.full(n_drivers, fill_value=default_m)
        sigma_init = np.full(n_drivers, fill_value=default_sigma)

        init_params = [baseline_init, alpha_init, m_init, sigma_init]

        return init_params

    if initializer == 'smart_start':
        baseline_init = initialize_baseline(driver_delays, driver_tt, acti_tt,
                                            T, lower=lower, upper=upper)
        alpha_init = []
        m_init = []
        sigma_init = []
        for p, delays in enumerate(driver_delays):
            delays = delays.data
            if delays.size == 0:
                alpha_init.append(- baseline_init)
                m_init.append(default_m)
                sigma_init.append(default_sigma)
            else:
                # compute Lebesgue measure of driver p supports
                s = compute_lebesgue_support(driver_tt[p], lower, upper)
                alpha_init.append(delays.size / s - baseline_init)
                m_init.append(np.mean(delays))
                if kernel_name == 'raised_cosine':
                    sigma = np.sqrt(
                        (np.pi**2 - 6)/(3 * np.pi**2)) * np.std(delays)
                    sigma_init.append(max(EPS, sigma))
                elif kernel_name == 'gaussian':
                    sigma_init.append(max(EPS, np.std(delays)))
    else:
        raise ValueError("Initializer method %s is unknown" % initializer)

    init_params = [baseline_init, alpha_init, m_init, sigma_init]

    return init_params


def compute_loss(loss_name, intensity, model, acti, T, driver=None):
    """

    Parameters
    ----------
    acti : torch.Tensor
        sparse tensor, 1 where there is an activation

    Returns
    -------
    XXX
    """
    if loss_name == 'log-likelihood':
        # first evaluate intensity integral
        nll = model.baseline * T
        n_driver_events = model.n_driver_events
        if driver is not None:
            n_driver_events = [this_driver.sum().item()
                               for this_driver in driver]
        for this_alpha, this_n in zip(model.alpha, n_driver_events):
            nll += this_alpha * this_n
        # negative log-likelihood
        return (nll - torch.log(intensity[np.where(acti)[0]]).sum())/T
    elif loss_name == 'MSE':
        return ((intensity ** 2).sum() * model.dt - 2 * (intensity[acti]).sum())/T
    else:
        raise ValueError(
            f"loss must be 'MLE' or 'log-likelihood', got '{loss_name}'"
        )


def optimizer(param, lr, solver='GD'):
    """

    Parameters
    ----------
    param : XXX

    lr : float
        learning rate

    solver : str
        solver name, possible values are 'GD', 'RMSProp', 'Adam', 'LBFGS'
        or 'CG'

    Returns
    -------
    XXX
    """
    if solver == 'GD':
        return optim.SGD(param, lr=lr)
    elif solver == 'RMSprop':
        return optim.RMSprop(param, lr=lr)
    elif solver == 'Adam':
        return optim.Adam(param, lr=lr, betas=(0.5, 0.999))
    elif solver == 'LBFGS':
        return optim.LBFGS(param, lr=lr)
    elif solver == 'CG':
        # XXX add conjugate gradient (not available in torch)
        raise ValueError(
            f"Conjugate gradient solver is not yet implemented."
        )
    else:
        raise ValueError(
            f"solver must be 'GD', 'RMSProp', 'Adam', 'LBFGS' or 'CG',"
            " got '{solver}'"
        )


def closure(model, driver_tt_train, acti_tt_train, optimizer):
    """

    """
    intensity = model(driver_tt_train)
    v_loss = compute_loss(
        model.loss_name, intensity, acti_tt_train, model.dt)
    optimizer.zero_grad()
    v_loss.backward()
    return v_loss


def training_loop(model, driver_tt, acti_tt, T=None, solver='RMSProp', step_size=1e-3,
                  max_iter=100, test=0.3, logging=True, device='cpu'):
    """Training loop for torch model.

    Parameters
    ----------
    true_params : list
        [mu_0, alpha_true, mu_true, sig_true]

    device : str
        'cpu' | 'cuda:0'  # cuda:x

    Returns
    -------
    XXX

    """

    driver_tt = check_tensor(driver_tt).to(device)
    acti_tt = check_tensor(acti_tt).to(device)

    if T is None:
        T = int(np.round(len(acti_tt)*model.dt))

    model = model.to(device)

    if test:
        # operates a train/test split
        n_test = int(np.round(test * driver_tt.shape[1]))
        driver_tt_train = driver_tt[:, :-n_test]
        driver_tt_test = driver_tt[:, -n_test:]

        acti_tt_train = acti_tt[:-n_test]
        acti_tt_test = acti_tt[-n_test:].to(torch.bool)
        T_test = int(np.round(T*test))
        T_train = T - T_test

    else:
        driver_tt_train = driver_tt
        acti_tt_train = acti_tt
        T_train = T

    driver_t = driver_tt_train.to(torch.bool)
    acti_tt_train = acti_tt_train.to(torch.bool)

    hist = []

    opt = optimizer(model.parameters(), step_size, solver=solver)

    start = time.time()
    for i in range(max_iter):
        print(f"Fitting model... {i/max_iter:6.1%}\r", end='', flush=True)

        if logging:
            hist.append(dict(
                baseline=model.baseline.detach().cpu().numpy().copy(),
                alpha=model.alpha.detach().cpu().numpy().copy(),
                m=model.m.detach().cpu().numpy(),
                time_loop=0 if i == 0 else time.time()-start
            ))
            if model.kernel_name != 'exponential':
                hist[-1].update(sigma=model.sigma.detach().cpu().numpy())

        if type(opt).__name__ == 'LBFGS':
            v_loss = closure(model, driver_tt_train, acti_tt_train, opt)
            opt.step(closure)
        else:
            opt.zero_grad()
            intensity = model(driver_tt_train)
            # v_loss = compute_loss(
            #     model.loss_name, intensity, acti_tt_train, model.dt, model,
            #     driver_tt_train)
            v_loss = model.compute_loss(
                intensity, acti=acti_tt_train, T=T_train,
                driver=driver_tt_train)

            if test:
                intensity_test = model(driver_tt_test)
            if logging:
                hist[-1].update(loss=v_loss.cpu().item())
                if test:
                    # loss_test = compute_loss(
                    #     model.loss_name, intensity_test,
                    #     acti_tt_test, model.dt, model, driver_tt_test).item()
                    loss_test = model.compute_loss(
                        intensity_test, acti=acti_tt_test, T=T_test,
                        driver=driver_tt_test).item()
                    hist[-1].update(loss_test=loss_test)
            v_loss.backward()
            opt.step()

        # projections
        model.alpha.data = model.alpha.data.clip(0)
        if model.kernel_name == 'raised_cosine':
            # ensure kernel stays in R+
            model.m.data = model.m.data.clip(0)
        if model.kernel_name != 'exponential':
            model.sigma.data = model.sigma.data.clip(0)

        # history
        # if logging:
        #     hist.append(dict(
        #         baseline=model.baseline.detach().cpu().numpy().copy(),
        #         alpha=model.alpha.detach().cpu().numpy().copy(),
        #         m=model.m.detach().cpu().numpy(),
        #         sigma=model.sigma.detach().cpu().numpy(),
        #         loss=v_loss.cpu().item(),
        #         time_loop=time.time()-start
        #     ))
        # if test:
        #     intensity_test = model(driver_tt_test)
        #     if logging:
        #         hist[-1].update(
        #             loss_test=compute_loss(model.loss_name, intensity_test,
        #                                    acti_tt_test, model.dt).item())

    end_time = time.time() - start
    print(f"Fitting model... done ({end_time:.3f} s.) ")

    est_params = {name: param.data
                  for name, param in model.named_parameters()
                  if param.requires_grad}

    print(f"Estimated parameters: {est_params}")

    res_dict = {'est_intensity': model(driver_tt_train).detach().cpu().numpy(),
                'est_kernel': model.kernels.detach().cpu().numpy(),
                'est_params': est_params,
                'compute_time': end_time}

    # compute final loss
    # v_loss = compute_loss(model.loss_name, intensity, acti_tt_train, model.dt,
    # model, driver_tt_train)
    v_loss = model.compute_loss(
        intensity, acti=acti_tt_train, T=T_train, driver=driver_tt_train)
    if test:
        intensity_test = model(driver_tt_test)
        res_dict.update(test_intensity=intensity_test.detach().cpu(),
                        n_test=n_test)

    if logging:  # save final parameters
        hist.append(dict(
            baseline=model.baseline.detach().cpu().numpy().copy(),
            alpha=model.alpha.detach().cpu().numpy().copy(),
            m=model.m.detach().cpu().numpy(),
            time_loop=end_time,
            loss=v_loss.cpu().item()
        ))
        if model.kernel_name != 'exponential':
            hist[-1].update(sigma=model.sigma.detach().cpu().numpy())
        if test:
            # loss_test = compute_loss(
            #     model.loss_name, intensity_test,
            #     acti_tt_test, model.dt, model, driver_tt_test).item()
            loss_test = model.compute_loss(
                intensity_test, acti=acti_tt_test, T=T_test,
                driver=driver_tt_test).item()
            hist[-1].update(loss_test=loss_test)

    res_dict['hist'] = hist

    return res_dict
