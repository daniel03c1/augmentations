import math
import torch
import torch.nn as nn
from torchvision import transforms

EPS = 1e-5


class SGCvD(nn.Module):
    """ simple and general controller (discrete version) """
    def __init__(self, 
                 bag_of_ops, 
                 n_transforms=4, 
                 n_layers=3,
                 h_dim=128,
                 dropout_p=0.5,
                 bins=21,
                 **kwargs):
        super(SGCvD, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.n_transforms = n_transforms
        self.h_dim = h_dim
        self.bins = bins

        ''' modules '''
        self.outdim = self.bins + self.n_transforms
        # self.means = torch.randn((1, self.outdim), requires_grad=True)
        # self.stds = torch.randn((1, self.outdim), requires_grad=True)
        self.register_parameter(
            name='means',
            param=nn.Parameter(torch.randn((1, self.n_ops, self.outdim)) 
                               / math.sqrt(self.outdim)))
        self.register_parameter(
            name='stds',
            param=nn.Parameter(torch.randn((1, self.n_ops, self.outdim)) 
                               / math.sqrt(self.outdim)))
        # self.means = self.means / math.sqrt(self.outdim)
        # self.stds = self.stds / math.sqrt(self.outdim)
        '''
        self.fixed_inputs = torch.tensor(
            torch.randn((1, h_dim)) / math.sqrt(h_dim),
            dtype=torch.float32,
            requires_grad=True)

        # middle
        self.n_layers = n_layers
        self.fcs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for i in range(n_layers)])
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm([h_dim]) for i in range(n_layers)])
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_p) for i in range(n_layers)])

        self.to_magnitude = nn.Linear(h_dim, self.n_ops * bins)
        self.to_prob = nn.Linear(h_dim, self.n_ops * n_transforms)
        '''

    def forward(self, x: torch.Tensor):
        n_samples = x.size(0)
        actions = self.stds*torch.randn((n_samples, self.n_ops, self.outdim)) \
                + self.means
        actions = actions.tanh()
        '''
        x = self.fixed_inputs.repeat(n_samples, 1)

        for i in range(self.n_layers):
            x = self.fcs[i](x)
            x = self.layernorms[i](x)
            x = x * x.sigmoid()
            x = self.dropouts[i](x)

        magnitudes = self.to_magnitude(x)
        magnitudes = magnitudes.reshape(-1, self.n_ops, self.bins)
        magnitudes = magnitudes.sigmoid()

        probs = self.to_prob(x)
        probs = probs.reshape(-1, self.n_ops, self.n_transforms).sigmoid()
        actions = torch.cat([magnitudes, probs], -1)
        '''
        return self.preprocess_actions(actions)

    def calculate_log_probs(self, actions):
        # layer*(n_ops + 1) actions
        # actions: [..., n_ops, bins+n_transforms]
        return actions.clamp(min=EPS).log()

    def calculate_entropy(self, actions):
        # actions: [..., layer, op, bins+1]
        # mean = actions.mean(0, keepdim=True)
        # return (actions - mean).abs()
        return self.stds.abs() # actions.std(0, keepdim=True)

    def preprocess_actions(self, actions):
        magnitudes = actions[..., :-self.n_transforms] + 1
        magnitudes /= magnitudes.sum(-1, keepdim=True)
        probs = actions[..., -self.n_transforms:] + 1
        probs /= probs.sum(-2, keepdim=True)

        return torch.cat([magnitudes, probs], -1)

    def decode_policy(self, action):
        # action = self.preprocess_actions(action.detach())
        return RandomApplyDiscrete(self.bag_of_ops, action, self.n_transforms)


class RandomApplyDiscrete(torch.nn.Module):
    def __init__(self, bag_of_ops, action, n_transforms):
        super().__init__()
        self.n_transforms = n_transforms
        self.magnitudes = action[..., :-n_transforms]
        self.probs = action[..., -n_transforms:]

        self.layers = [bag_of_ops[i](self.magnitudes[i])
                       for i in range(bag_of_ops.n_ops)]

    def forward(self, image):
        opers = self.probs.multinomial(1).squeeze()
        for i in range(self.n_transforms):
            image = self.layers[opers[i]](image)
        return image


if __name__ == '__main__':
    from discrete_transforms import transforms as bag
    from utils import get_default_device

    brain = SGCvD(bag, n_transforms=3)
    print(sum([p.numel() for p in brain.parameters()]))

    x = torch.zeros(2)
    actions = brain(x)
    probs = brain.calculate_log_probs(actions).exp()

    policy = brain.decode_policy(actions[0])
    y = torch.rand((32, 3, 32, 32))
    for i in range(3):
        print(policy(y).size())

