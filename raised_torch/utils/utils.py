import torch


def check_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    return x
