import numpy as np
import torch 
import itertools
import pandas as pd


from raised_torch.model import Model
from raised_torch.simu_pp import simu
from raised_torch.solver import initialize, optimizer, training_loop
from raised_torch.utils.utils import check_tensor



def procedure(t, simu_params, kernel_params, kernel_name='raised_cosine', device='cpu', seed=None):
    """ 
    """
    if seed is None:
        seed = 0
    #simulate data
    kernel_value, intensity_value, driver_tt, driver, acti_tt, acti = simu(
                                                            kernel_params[0], 
                                                            kernel_params[1], 
                                                            kernel_params[2],
                                                            kernel_params[3],
                                                            simu_params=[simu_params[0],
                                                            simu_params[1], simu_params[2]],
                                                            isi=simu_params[3],
                                                            kernel_name=kernel_name,
                                                            seed=seed, 
                                                            plot_intensity=False,   
                                                            lower=kernel_params[4],
                                                            upper=kernel_params[5])
    
    
    #Smart init of parameters
    init_params = initialize(driver_tt, acti_tt, simu_params[0], 
                             initializer='smart_start', lower=0, upper=0.8,
                             kernel_name=kernel_name)

    hist_dic = dict(
             simu_param=simu_params,
             kernel_param=kernel_params,
             seed=seed
             #kernel=kernel_value,
             #intensity=intensity_value,
             #driver_tt=driver_tt,
             #driver=driver,
             #acti_tt=acti_tt,
             #acti=acti
    )
    
    baseline_init, alpha_init, m_init, sigma_init = init_params
    
    model = Model(t, 
                  baseline_init, 
                  alpha_init, 
                  m_init,
                  sigma_init,
                  lower=kernel_params[4],
                  upper=kernel_params[5])
    
    res_dic = training_loop(model,  driver,
                             acti, simu_params[4], 
                             simu_params[5], 
                             simu_params[6],
                             test=0.2,
                             device=device)
    
    wanted_keys = ['est_params', 'n_test']
                        
    hist_dic.update(dict((k, res_dict[k]) for k in wanted_keys if k in hist_dic))

    return hist_dic

def benchmark_synthetic(n_simu, simu_params, kernel_params, kernel_name, device):
    """
    """
    hist = []
    dt = 1 / simu_params['L'][0]
    t = torch.arange(0, 1, dt)

    #Combinations of simulation and kernel parameters
    combs_simu = list(itertools.product(*list(simu_params.values())))
    combs_kernel = list(itertools.product(*list(kernel_params.values())))

    for i in range(n_simu):
        for l in range(len(combs_simu)):
            for j in range(len(combs_kernel)):
        
                hist_dic = procedure(t, simu_params=combs_simu[l], 
                             kernel_params=combs_kernel[j],
                             kernel_name=kernel_name,
                             device=device,
                             seed=i)   
                hist.append(hist_dic)   

        
    return hist