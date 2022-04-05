##########################################
# Useful functions for optimization
##########################################

import numpy as np
import torch
import torch.optim as optim
import time

from kernels import raised_cosine_kernel


def compute_loss(loss_name, intensity, acti_t, dt):
    """
    """
    T = len(intensity) * dt
    if loss_name == 'log-likelihood':
        # negative log-likelihood
        return (intensity.sum() * dt - torch.log(intensity[acti_t]).sum()) / T
    elif loss_name == 'MSE':
        return ((intensity ** 2).sum() * dt - 2 * (intensity[acti_t]).sum()) / T
    else:
        raise ValueError(
            f"loss must be 'MLE' or 'log-likelihood', got '{loss_name}'"
        )

def optimizer(param, lr, solver='SGD'):
    if solver == 'GD':
        return optim.SGD(param, lr=lr)
    elif solver == 'RMSprop':
        return optim.RMSprop(param, lr=lr)
    elif solver == 'Adam':
        return optim.Adam(param, lr=lr, betas=(0.5, 0.999))
    else:
        raise ValueError(
            f"solver must be 'GD' | 'RMSProp' or 'Adam' ', got '{solver}'"
        )

def training_loop(model, optimizer, driver_tt, acti_tt,  max_iter=100,  test=0.3):
    """Training loop for torch model.

        Parameters
    ----------
    true_params : list
        [mu_0, alpha_true, mu_true, sig_true]"""

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
        intensity = model(driver_torch)
        v_loss = compute_loss(model.loss_name, intensity, acti_t, model.dt)
        v_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        model.weights.data[1] = max(0, model.weights.data[1]) # alpha
        model.weights.data[3] = max(0, model.weights.data[3]) # sigma

        pobj.append(v_loss.item()) 

        if test:
            intensity_test = model(driver_torch_test)

            pval.append(compute_loss(model.loss_name, intensity_test, acti_t_test, model.dt).item())

    print(f"Fitting model... done ({np.round(time.time()-start)} s.) ")
    print(f"Estimated parameters: {np.array(model.weights.data)}")


    res_dict = {'est_intensity': np.array(intensity.detach()),
                'est_kernel': np.array(model.kernel.detach()),
                'pobj': pobj,
                'est_params': model.weights.data}
    if test:
        res_dict.update(test_intensity=np.array(intensity_test.detach()),
                        pval=pval, n_test=n_test)

    return res_dict