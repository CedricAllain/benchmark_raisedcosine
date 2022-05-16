import numpy as np
import torch
import matplotlib.pyplot as plt

from raised_torch.kernels import compute_kernels

from trunc_norm_kernel.model import TruncNormKernel

# %% Compare torch and dripp kernels
plt.rcParams.update(plt.rcParamsDefault)
figsize = (5, 3)

# m, sigma = 0.9, 0.1
m, sigma = 0.9, 0.1
lower, upper = 0, 1

# plot true kernel
L = 1000
kernel_cont = TruncNormKernel(
    lower, upper, m=m, sigma=sigma, use_dis=False)
t = np.arange(lower-0.1, upper+0.1+1/L, 1/L)
fig = plt.figure(figsize=figsize)
plt.plot(t, kernel_cont(t))
plt.xlim(t.min(), t.max())
plt.xlabel("t (s)")
plt.title(f"True kernel: m={m}, sigma={sigma}")
plt.show()

diff_torch = []
diff_dripp = []
t_max = []
step = 10
L_list = np.arange(50, 1_000+step, step=step)
for L in L_list:
    kernel_dis = TruncNormKernel(
        lower, upper, m=m, sigma=sigma, sfreq=L, use_dis=True)
    dt = 1/L
    kernel_support = torch.arange(lower, upper+dt, dt)
    kernels_torch = compute_kernels(
        kernel_support, [1], [m], [sigma], "gaussian", lower, upper, dt)
    t = kernel_support.numpy()
    this_diff = kernels_torch[0].numpy() - kernel_dis(t)
    diff_torch.append(np.abs(this_diff).max())
    t_max.append(np.argmax(np.abs(this_diff)) / L)
    # this_diff_dripp = np.abs(kernel_cont(t) - kernel_dis(t))
    # diff_dripp.append(this_diff_dripp.max())

_, ax = plt.subplots(figsize=figsize)
ax.plot(L_list, diff_torch, color="blue")
ax.set_xlim(min(L_list), max(L_list))
ax.set_xlabel('L')
ax.set_ylabel("abs max diff", color="blue")
ax2 = ax.twinx()
ax2.plot(L_list, t_max, color="red")
ax2.hlines(m, min(L_list), max(L_list), linestyles='--', color="red",
           alpha=0.5, label=f'm = {m}')
ax2.set_ylim(lower, upper)
ax2.set_ylabel("time of max diff (s)", color="red")
ax2.legend()
plt.title("max abs diff between torch and dripp kernel")
plt.show()

# plt.plot(L_list, diff_dripp)
# plt.title("max abs diff between dripp kernels")
# plt.show()

print(f"max abs diff for L = 150: {np.array(diff_torch)[L_list==150][0]:.5f}")
# Plot 3 kernels
L = 150
dt = 1/L
kernel_support = torch.arange(lower, upper+dt, dt)
kernels_torch = compute_kernels(
    kernel_support, [1], [m], [sigma], "gaussian", lower, upper, dt)
fig = plt.figure(figsize=figsize)
plt.plot(kernel_support, kernels_torch[0], label='torch')
t_cont = np.arange(lower, upper+1/1000, 1/1_000)
plt.plot(t_cont, kernel_cont(t_cont), label='cont')
kernel_dis = TruncNormKernel(
    lower, upper, m=m, sigma=sigma, sfreq=L, use_dis=True)
t_dis = kernel_support.numpy()
plt.plot(t_dis, kernel_dis(t_dis), label='dis')
plt.title(f"3 kernels for L = {L}")
plt.legend()
plt.show()

# %%
