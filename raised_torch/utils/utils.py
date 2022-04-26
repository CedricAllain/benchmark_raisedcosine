import torch


def check_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    return x


def kernel_intensity(baseline, driver, kernels, L=100):
    """
    Parameters
    ----------
    baseline : float
        baseline intensity parameter

    driver : sparse vector where non nul values correspond to driver timestamps

    kernel : vector of length L, the kernel values over its support

    L : int
        kernel times

    Returns
    -------
    intensity : torch.Tensor
    """
    intensity = baseline + torch.conv_transpose1d(
        torch.Tensor(driver)[None], kernels[:, None])[0, 0, :-L+1]
    return intensity.clip(0)
