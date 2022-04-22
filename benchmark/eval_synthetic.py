# %%
import numpy as np
import torch as nn

from model import Model
from simu_pp import simu

# params = nn.tensor([[1, np.nan], # baseline
#                     [0.7, 0.5],  # alpha
#                     [0.4, 0.6],  # m
#                     [0.4, 0.2]]) # sigma

params = nn.tensor([[1], # baseline
                    [0.7],  # alpha
                    [0.4],  # m
                    [0.4]]) # sigma

seed = 0
T = 10_000
L = 100
dt = 1 / L
p_task = 0.6
t = nn.arange(0, 1, dt)

model = Model(t, params)

# %% Simulate data
# simulate data
kernel_value, intensity_value, driver_tt, driver, acti_tt, acti = simu(
    params, simu_params=[T, L, p_task], seed=seed, plot_intensity=False)
# %%
