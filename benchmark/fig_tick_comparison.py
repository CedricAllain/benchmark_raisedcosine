"""
Plot, using tick methods, the computation time as a function of the problem
size, to show that with non parametrized kernel, tick is slow with
discretization.
"""
# %% Packages
import numpy as np
import matplotlib.pyplot as plt
import time

from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp,
                         HawkesEM)
from tick.base import TimeFunction
from tick.plot import plot_hawkes_kernels, plot_point_process

from raised_torch.model import Model
from raised_torch.solver import initialize, training_loop

# %%
T = 10_000
L = 100
max_iter = 100
N_JOBS = 4
dt = 1/L
lower, upper = 0, 5
t_values = np.arange(lower, upper, dt)
baseline = 0.5
alpha = 0.5
gamma = 0.7
assert np.exp(-gamma * L) < 1
exp_kernel = alpha * gamma * np.exp(-gamma * t_values)

plt.plot(t_values, exp_kernel, label="True kernel")
plt.legend()
plt.title("True exponential kernel")
plt.xlabel('t')
plt.show()

# %% Simulate data

hawkes = SimuHawkes(baseline=np.array([baseline]), end_time=T, verbose=False,
                    seed=42)
hawkes.set_kernel(0, 0, HawkesKernelExp(alpha, gamma))
hawkes.track_intensity(dt)
hawkes.simulate()
intensity = hawkes.tracked_intensity
intensity_times = hawkes.intensity_tracked_times
plot_point_process(hawkes, n_points=50000, t_min=0, t_max=20)
plt.show()
# %% Learn with tick
em = HawkesEM(upper, kernel_size=(upper*L), n_threads=N_JOBS, verbose=False,
              tol=1e-3, max_iter=max_iter)

start = time.time()
print(f"Fitting model...\r", end='', flush=True)
em.fit(hawkes.timestamps)
end_time = time.time() - start
print(f"Fitting model... done ({np.round(end_time)} s.) ")

fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)
plt.ylim([0, None])
plt.xlim([lower, upper])
plt.show()
# %% Learn with torch
solver = "RMSprop"
kernel_name = "exp"
loss_name = 'log-likelihood'
# initialize parameters
driver_tt = hawkes.timestamps
init_params = initialize(driver_tt, acti_tt, T, initializer='smart_start',
                         lower=lower, upper=upper,
                         kernel_name=kernel_name)
baseline_init, alpha_init, m_init, sigma_init = init_params

model_raised = Model(t_values, baseline_init, alpha_init, m_init, sigma_init, dt,
                     kernel_name=kernel_name, loss_name=loss_name,
                     lower=lower, upper=upper)
res_dict = training_loop(model_raised, driver, acti, solver=solver,
                         step_size=1e-3, max_iter=max_iter, test=False,
                         logging=True, device='cpu')
