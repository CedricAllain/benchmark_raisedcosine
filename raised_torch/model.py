##########################################
# Define the class of intensity function
##########################################

#from scipy.optimize import check_grad
#from scipy.optimize import fmin_l_bfgs_b
#import itertools
#import pandas as pd
#from joblib import Parallel, delayed
#import pickle

import numpy as np
from tqdm import tqdm
import time
import torch
from torch import nn

from kernels import raised_cosine_kernel





class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, t, params, dt, 
                 kernel_name, loss_name):
        
        super().__init__()

        self.weights = nn.Parameter(params)
        self.t = t
        self.dt = dt
        self.L = len(self.t)       
        self.kernel_name = kernel_name
        self.loss_name = loss_name

        
    def forward(self, driver_tt_torch):
        """Function to be optimised (the intensity).,
        """

        mu_0, alpha, mu, sig = self.weights

        if self.kernel_name == 'gaussian':
            self.kernel = truncated_gaussian_kernel(self.t, self.weights, self.dt)           
        elif self.kernel_name == 'raised_cosine':
            self.kernel = raised_cosine_kernel(self.t, self.weights, self.dt)

        intensity = mu_0 + torch.conv_transpose1d(driver_tt_torch[None, None], 
                                                self.kernel[None, None],
                                                )[0, 0, :-self.L+1]

        return intensity










