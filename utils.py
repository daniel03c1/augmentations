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


'''
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


x = torch.rand(4, 3, 32, 32)
model = torchvision.models.resnet18(pretrained=False)
avg = model.avgpool.register_forward_hook(get_activation('avgpool'))

out = model(x)
print(out)
print(model.fc(activation['avgpool'].squeeze()))

avg.remove() # detach hooks
'''

