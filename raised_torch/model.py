##########################################
# Define the class of intensity function
##########################################
from torch import nn


from .kernels import compute_kernels
from .utils.utils import check_tensor, kernel_intensity


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.

    Parameters
    ----------
    t : XXX

    params : 2darray-like
        ex.: params = nn.tensor([[1, np.nan],  # baseline
                                 [0.7, 0.5],   # alpha
                                 [0.4, 0.6],   # m
                                 [0.4, 0.2]])  # sigma

    dt : XXX

    kernel_name : str, 'gaussian' | 'raised_cosine'
        name of

    loss_name : XXX

    """

    def __init__(self, t, baseline, alpha, m, sigma, dt=1/100,
                 kernel_name='raised_cosine', loss_name='log-likelihood',
                 lower=None, upper=None):

        super().__init__()

        self.kernel_name = kernel_name

        self.baseline = nn.Parameter(check_tensor(baseline))
        self.alpha = nn.Parameter(check_tensor(alpha))
        if (self.kernel_name == 'gaussian') or (self.kernel_name == 'exponential'):
            self.m = nn.Parameter(check_tensor(m))
        elif self.kernel_name == 'raised_cosine':
            # reparametrazion, u = m - sigma
            self.m = nn.Parameter(check_tensor(m) - check_tensor(sigma))
        else:
            raise ValueError(
                "kernel_name must be 'gaussian' | 'raised_cosine' | 'exponential',"
                f" got '{self.kernel_name}'"
            )
        if sigma is not None:
            self.sigma = nn.Parameter(check_tensor(sigma))
        else:
            self.sigma = None

        self.lower = lower
        self.upper = upper
        self.register_buffer('t', t)

        # compute initial kernels
        self.kernels = compute_kernels(
            self.t, self.alpha, self.m, self.sigma, self.kernel_name,
            self.lower, self.upper)

        self.dt = dt
        self.L = len(self.t)
        self.loss_name = loss_name

    def forward(self, driver):
        """Function to be optimised (the intensity).,

        Parameters
        ----------
        driver : 2d torch.Tensor
            size n_drivers, n_times

        Returns
        -------
        intensity : XXX
        """

        # compute updated kernels
        self.kernels = compute_kernels(
            self.t, self.alpha, self.m, self.sigma, self.kernel_name,
            self.lower, self.upper)

        return kernel_intensity(self.baseline, driver, self.kernels, self.L)
