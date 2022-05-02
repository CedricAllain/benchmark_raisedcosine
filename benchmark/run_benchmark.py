import numpy as np
import torch 
import itertools
import pandas as pd
import os.path as op
import os

from raised_torch.model import Model
from raised_torch.simu_pp import simu
from raised_torch.solver import initialize, optimizer, training_loop
from raised_torch.utils.utils import check_tensor

from benchmark.benchmark_ import procedure, benchmark_synthetic

############################
# Simu parameters
############################
n_simu = 10


simu_params = {'T': [1000],
               'L': [100],
               'p_tasks': [0.1, 0.3],
               'isi': [0.7, 1],
               'solver': ['Adam', 'RMSprop', 'GD'],
               'step_size': [1e-3],
               'max_iter': [1000]
               }

############################
# Kernel parameters
############################
n_drivers = 1
kernel_name = 'raised_cosine'

alpha = [0.5, 0.6, 0.7]
m = [0.4, 0.7, 0.5]
sigma = [0.05, 0.1, 0.2]




alpha_all = list(itertools.combinations(alpha, n_drivers))
m_all = list(itertools.combinations(m, n_drivers))
sigma_all = list(itertools.combinations(sigma, n_drivers))


kernel_params = {'baseline': [1.5],
               'alpha': alpha_all,
               'm': m_all,
               'sigma': sigma_all,
               'lower': [0], 
               'upper': [800e-3], 
               }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


############################
#Simulation
############################
results = benchmark_synthetic(n_simu, simu_params, kernel_params, kernel_name, device)


############################
#Save results
############################

if not op.exists("results"):
	os.mkdir("results")
	
save_results_path = op.join("results/results_{kernel}_{driver}".format(
    kernel=kernel_name, driver=n_drivers))

results_dataframe = pd.DataFrame.from_dict(results)

results_dataframe.pickle(save_results_path)