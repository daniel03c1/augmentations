import math
import numpy as np
import torch
import torchvision


class RunningStats:
    # Welford Algorithm
    def __init__(self):
        self.n = 0
        self.mu = None # mean
        self.sig2 = None # var

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        # init
        if self.n == 1:
            self.mu = x * 0
            self.sig2 = x * 0

        # calculate
        last_mu = self.mu
        self.mu += (x - last_mu) / self.n
        self.sig2 += ((x-last_mu)*(x-self.mu) - self.sig2) / self.n

    def mean(self):
        return self.mu if self.n else 0.

    def var(self):
        return self.sig2 if self.n > 1 else 0.

    def std(self):
        if isinstance(self.sig2, (int, float)):
            return math.sqrt(self.var())
        elif isinstance(self.sig2, np.ndarray):
            return np.sqrt(self.var())
        elif isinstance(self.sig2, torch.Tensor):
            return torch.sqrt(self.var())
        else:
            raise ValueError(f'unsupported type {type(self.sig2)}')


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_resize(src, des_dim):
    return src.matmul(lin_mapping_matrix(src.size(-1), 
                                         des_dim, 
                                         device=src.device))


def lin_mapping_matrix(src_dim, des_dim, device=None):
    result = torch.linspace(0, src_dim-1, des_dim, device=device)
    result = result.repeat(src_dim, 1) \
           - torch.arange(0, src_dim, device=device).unsqueeze(-1)
    result = result.abs()
    result = torch.where(result <= 1, 1-result, torch.zeros_like(result))
    return result


def standard_normalization(x, eps=1e-8):
    return (x - x.mean()) / x.std().clamp(min=eps)

