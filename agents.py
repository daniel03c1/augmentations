import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from utils import *
from valuator import Valuator

DEVICE = get_default_device()
EPS = 1e-8


class BasicAgent:
    def __init__(self,
                 bag_of_ops,
                 n_transforms=2,
                 min_bins=3,
                 max_bins=11,
                 M=8,
                 mem_maxlen=1,
                 batch_mem_maxlen=1,
                 lr=0.00035,
                 grad_norm=0.5,
                 epsilon=0.2,
                 ent_coef=1e-5,
                 **args):
        self.device = get_default_device()
        n_ops = bag_of_ops.n_ops

        self.controller = BasicController(
            n_ops, n_transforms, min_bins, device=self.device).to(self.device)
        self.valuator = Valuator(
            [n_ops, n_transforms+max_bins]).to(self.device)

        self.bag_of_ops = bag_of_ops
        self.n_ops = n_ops
        self.n_transforms = n_transforms
        self.min_bins = min_bins
        self.max_bins = max_bins

        self.c_optimizer = optim.Adam(self.controller.parameters(), 
                                      lr=lr)
        self.l1loss = nn.L1Loss(reduction='none')
        self.v_optimizer = optim.Adam(self.valuator.parameters(), 
                                      weight_decay=1e-5)
        self.v_criterion = nn.L1Loss()
        self.grad_norm = grad_norm

        self.sample_memory = deque(maxlen=mem_maxlen * M)
        self.batch_memory = deque(maxlen=batch_mem_maxlen)
        self.M = M
        self.epsilon = epsilon
        self.ent_coef = ent_coef

    def act(self, n_samples, enable_dropout=True):
        self.controller.eval()
        if enable_dropout:
            for m in self.controller.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        return self.controller(n_samples)

    def cache(self, action, reward):
        self.sample_memory.append((action.to(self.device).detach(),
                                   reward.to(self.device).detach()))

    def cache_batch(self, actions, rewards):
        for i in range(actions.size(0)):
            self.cache(actions[i], rewards[i])

        actions = self.preprocess_actions(actions, resize=True)
        rewards = standard_normalization(rewards)

        self.batch_memory.append((actions.to(self.device).detach(),
                                  rewards.to(self.device).detach()))

    def learn(self, n_steps=1):
        self.train_valuator()

        self.controller.train()
        for i in range(n_steps):
            self.c_optimizer.zero_grad()

            old_actions, _ = self.recall(len(self.sample_memory))
            old_actions.extend(
                [torch.rand(*old_action.size(), device=old_action.device)
                 for old_action in old_actions]) # add random data
            processed = torch.stack(
                [self.preprocess_actions(old_action, resize=True)
                 for old_action in old_actions])
            rewards = self.valuator(processed).squeeze(-1)
            rewards = standard_normalization(rewards)
            
            rank = rewards.argsort()
            best = rank[-int(len(self.sample_memory)*0.5):].tolist()
            worst = rank[:int(len(self.sample_memory)*0.5)].tolist()
            old_actions = [old_action for i, old_action in enumerate(old_actions)
                           if i in best] \
                        + [old_action for i, old_action in enumerate(old_actions)
                           if i in worst]
            rewards = torch.cat([rewards[best], rewards[worst]])

            old_actions = torch.stack(
                [self.preprocess_actions(old_action, normalize=False,
                                         resize=True, des_dim=self.get_bins())
                 for old_action in old_actions])
            actions = self.controller(old_actions.size(0))

            old_actions = self.preprocess_actions(old_actions)
            actions = self.preprocess_actions(actions)

            # Dist
            loss = self.l1loss(actions, old_actions)
            loss *= rewards.view(-1, *([1]*(len(loss.shape)-1)))

            loss -= self.ent_coef \
                  * self.controller.calculate_entropy(actions).clamp(min=1e-8).sqrt()

            torch.mean(loss).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                           self.grad_norm)
            self.c_optimizer.step()

        return loss # the last loss

    def recall(self, n_samples=None, preprocess=False, *args, **kwargs):
        if n_samples is None:
            n_samples = self.M
        batch = random.sample(self.sample_memory, n_samples)
        actions = []
        rewards = []
        for action, reward in batch:
            if preprocess:
                action = self.preprocess_actions(action, *args, **kwargs)
            actions.append(action)
            rewards.append(reward)
        return actions, torch.stack(rewards)

    def train_valuator(self, tolerance=0.05):
        self.valuator.train()
        epochs = len(self.batch_memory)

        while True:
            total_loss = 0

            for _ in range(epochs):
                actions, rewards = random.sample(self.batch_memory, 1)[0]

                self.v_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.valuator(actions).squeeze(-1)
                    loss = self.v_criterion(standard_normalization(ys_hat), rewards)
                    loss.backward(retain_graph=True)
                    self.v_optimizer.step()

                    wrong = ys_hat.argsort() != rewards.argsort()
                    total_loss += wrong.float().mean() / epochs

            if total_loss < tolerance:
                break
        self.valuator.eval()

    def preprocess_actions(self, 
                           action, 
                           normalize=True, 
                           resize=False, 
                           des_dim=None):
        action = action.clamp(min=EPS)
        magnitudes = action[..., :-self.n_transforms]
        probs = action[..., -self.n_transforms:]

        if des_dim is None:
            des_dim = self.max_bins
        if resize:
            magnitudes = linear_resize(magnitudes, des_dim)
        if normalize:
            magnitudes /= magnitudes.sum(-1, keepdim=True)
            probs /= probs.sum(-2, keepdim=True)

        return torch.cat([magnitudes, probs], -1)

    def decode_policy(self, action, *args, **kwargs):
        action = self.preprocess_actions(action.detach(), *args, **kwargs)
        return DiscreteRandomApply(self.bag_of_ops, action, self.n_transforms)

    def get_bins(self):
        return self.controller.bins

    def reset_bins(self, bins):
        self.controller.reset_bins(bins)


class BasicController(nn.Module):
    """ simple and general controller (discrete version) """
    def __init__(self, 
                 n_ops, 
                 n_transforms=4, 
                 bins=3,
                 n_layers=2,
                 h_dim=256,
                 dropout_p=0.4,
                 device=None,
                 **kwargs):
        super(BasicController, self).__init__(**kwargs)
        self.n_ops = n_ops
        self.n_transforms = n_transforms
        self.h_dim = h_dim
        self.bins = bins

        self.device = device

        ''' modules '''
        self.outdim = self.bins + self.n_transforms
        self.fixed_inputs = torch.randn((1, h_dim), 
                                        requires_grad=True,
                                        device=device) \
                          / math.sqrt(h_dim)

        # middle
        self.n_layers = n_layers
        self.fcs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for i in range(n_layers)])
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(h_dim) for i in range(n_layers)])
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_p) for i in range(n_layers)])

        self.to_magnitude = nn.Linear(h_dim, self.n_ops * bins)
        self.to_prob = nn.Linear(h_dim, self.n_ops * n_transforms)

    def forward(self, n_samples):
        x = self.fixed_inputs.repeat(n_samples, 1)

        for i in range(self.n_layers):
            x = self.fcs[i](x)
            x = self.batchnorms[i](x)
            x = x * x.sigmoid()
            x = self.dropouts[i](x)

        magnitudes = self.to_magnitude(x)
        magnitudes = magnitudes.reshape(-1, self.n_ops, self.bins)

        probs = self.to_prob(x)
        probs = probs.reshape(-1, self.n_ops, self.n_transforms)
        return torch.cat([magnitudes, probs], -1).sigmoid()

    def calculate_log_probs(self, actions):
        return actions.clamp(min=EPS).log()

    def calculate_entropy(self, actions):
        return (actions - actions.mean(0, keepdim=True)).abs()

    def reset_bins(self, bins):
        with torch.no_grad():
            weight = self.to_magnitude.weight # (n_ops * bins, h_dim)
            bias = self.to_magnitude.bias # (n_ops * bins)

            # weight update
            weight = weight.reshape(self.n_ops, self.bins, -1)
            weight = weight.transpose(-2, -1)
            weight = linear_resize(weight, bins)
            weight = weight.transpose(-2, -1)
            weight = weight.reshape(-1, weight.size(-1))
            self.to_magnitude.weight = nn.parameter.Parameter(
                weight, requires_grad=True)

            # bias update
            bias = bias.reshape(self.n_ops, self.bins)
            bias = linear_resize(bias, bins)
            bias = bias.reshape(-1)
            self.to_magnitude.bias = nn.parameter.Parameter(
                bias, requires_grad=True)

            self.bins = bins


class DiscreteRandomApply(torch.nn.Module):
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

    agent = BasicAgent(bag,
                       n_transforms=2,
                       min_bins=3,
                       max_bins=11,
                       M=8,
                       mem_maxlen=1,
                       batch_mem_maxlen=1,
                       lr=0.00035,
                       grad_norm=0.5,
                       epsilon=0.2,
                       ent_coef=1e-5)
    actions = agent.act(8, enable_dropout=True)
    rewards = torch.rand(8)
    print(agent.act(1).size())
    agent.cache_batch(actions, rewards)
    agent.learn(n_steps=32)
    policy = agent.decode_policy(actions[0], resize=False)

    xs = torch.rand(8, 3, 32, 32)
    for i in range(3):
        print(policy(xs).size())

    ct = BasicController(bag.n_ops, n_transforms=4, bins=3)
    print(ct(8)[0])
    ct.reset_bins(4)
    print(ct(8)[0])
    ct.reset_bins(5)
    print(ct(8)[0])

