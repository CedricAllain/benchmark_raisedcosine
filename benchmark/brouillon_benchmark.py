# %%
import numpy as np
import torch 


from raised_torch.model import Model
from raised_torch.simu_pp import simu
from raised_torch.solver import initialize, optimizer, training_loop

from trunc_norm_kernel.model import TruncNormKernel
from trunc_norm_kernel.optim import em_truncated_norm
from trunc_norm_kernel.simu import simulate_data


seed = 0

############################
# Simu parameters
############################

n_simu = 10
N_JOBS = 10
#N_DRIVERS = [3]
#rep_drivers = len(N_DRIVERS)
reparam = True
test = True

T = 1000; L = 100; p_task = 0.6
simu_params = [T, L, p_task]
dt = 1 / L

############################
# model parameters
############################
baseline = torch.tensor(1.5)
alpha = torch.tensor([0.7, 0.8, 0.9])
mu = torch.tensor([0.4, 0.4, 0.4])
s = torch.tensor([0.1, 0.2, 0.3])
params = [baseline, alpha, mu, s]


############################
#optim parameters
############################
SOLVER = ['Adam', 'RMSprop', 'GD']
step_size = 1e-3
max_iter = 200

############################


kernel_values = []; intensity_values = []
driver_tts = []; drivers = []
acti_tts = []; actis = []
init_params_ = []

t = torch.arange(0, 1, dt)

############################
# simulate data and init params model
############################
for l in range(n_simu):
        #simulate data
        kernel_value, intensity_value, driver_tt, driver, acti_tt, acti = simu(
            params, simu_params=simu_params, seed=seed, plot_intensity=False)
        
        # Smart init of parameters
        init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start', lower=0, upper=0.8,
               kernel_name='raised_cosine')

        kernel_values.append(kernel_value)
        intensity_values.append(intensity_value)
        driver_tts.append(driver_tt)
        drivers.append(driver)
        acti_tts.append(acti_tt)
        actis.append(acti)
        
        # Smart init of parameters
        if reparam:
            init_params_.append(init_params)
        else:
            baseline_init, alpha_init, m_init, sigma_init = init_params
            init_params = [baseline_init, alpha_init, m_init - sigma_init , sigma_init]
            init_params_.append(init_params)
        
        #init model
        model_raised = Model(t, init_params, reparam=reparam)
        for opt_algo in SOLVER:
            opt =  optimizer(model_raised.parameters(), step_size, opt_algo)
            res_dict = training_loop(model_raised, opt, driver, acti, max_iter, test)
############################    




############################ 