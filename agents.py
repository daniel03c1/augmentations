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
        self.v_criterion = nn.BCELoss(reduction='none') # L1Loss()
        self.grad_norm = grad_norm

        self.sample_memory = deque(maxlen=mem_maxlen * M)
        self.batch_memory = deque(maxlen=batch_mem_maxlen)
        self.M = M
        self.ent_coef = ent_coef

        # mask
        self.mask = torch.arange(M)
        self.mask  = (self.mask - self.mask.unsqueeze(-1) > 0).float() \
                                                              .to(self.device)
        self.mask = self.mask / self.mask.sum()

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

    def learn(self, n_steps=1, max_step=0.01):
        self.train_valuator()

        self.controller.train()
        self.valuator.zero_grad()

        for i in range(n_steps):
            self.c_optimizer.zero_grad()

            actions = self.controller(self.M)

            # calculate target actions
            with torch.set_grad_enabled(True):
                target_actions = actions.detach()
                target_actions.requires_grad = True

                scores = self.valuator(
                    self.preprocess_actions(target_actions, resize=True))
                (-scores.mean()).backward()

                action_grads = target_actions.grad.data
                action_grads = action_grads.clamp(min=-max_step, max=max_step)

                target_actions = target_actions.detach() + action_grads.detach()
                target_actions = target_actions.clamp(min=0, max=1)
                self.valuator.zero_grad()

            ents = self.controller.calculate_entropy(actions)
            # probs = self.controller.calculate_probs(target_actions)
            # loss = - probs.mean() - self.ent_coef * ents.mean()
            log_probs = self.controller.calculate_log_probs(target_actions)
            loss = - log_probs.mean() - self.ent_coef * ents.mean()

            loss.backward(retain_graph=True)
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

                ys_comb = rewards - rewards.unsqueeze(-1) > 0
                ys_comb = ys_comb.float().to(self.device)

                self.v_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.valuator(actions).squeeze(-1)

                    ys_hat_comb = (ys_hat - ys_hat.unsqueeze(-1)).sigmoid()
                    loss = self.v_criterion(ys_hat_comb, ys_comb)
                    loss = (loss * self.mask).sum()
                    # loss = self.v_criterion(standard_normalization(ys_hat), 
                    #                         rewards)
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
                 eps=1e-6,
                 device=None,
                 **kwargs):
        super(BasicController, self).__init__(**kwargs)
        self.n_ops = n_ops
        self.n_transforms = n_transforms
        self.h_dim = h_dim
        self.bins = bins

        self.device = device

        ''' modules '''
        outdim = self.bins + self.n_transforms
        self.means = nn.parameter.Parameter(
            torch.randn((1, self.n_ops, outdim), device=device),
            requires_grad=True)
        self.stds = nn.parameter.Parameter(
            torch.randn((1, self.n_ops, outdim), device=device),
            requires_grad=True)

        # for gaussian distribution
        self.eps = eps
        self.pi2 = 2 * math.pi
        self.logprobs_const = - math.log(math.sqrt(self.pi2))
        self.ent_const = 0.5 + 0.5 * math.log(self.pi2)

    def forward(self, n_samples):
        rand = torch.randn((n_samples, self.n_ops, self.bins+self.n_transforms),
                           device=self.device)
        return (self.means + rand).sigmoid()

    def calculate_log_probs(self, actions):
        # reverse sigmoid
        actions = -torch.log(
            1/actions.clamp(min=self.eps, max=1-self.eps) - 1)

        stds = self.stds.clamp(min=self.eps)
        # probs = torch.exp(-torch.square((actions-self.means)/stds)/2) \
        #       / (math.sqrt(self.pi2) * stds)
        # return probs
        log_probs = - torch.square((actions-self.means)/stds)/2 \
                  - stds.log() + self.logprobs_const
        return log_probs

    def calculate_entropy(self, actions):
        return self.ent_const + self.stds.clamp(min=self.eps).log()

    def reset_bins(self, bins):
        with torch.no_grad():
            new_means = torch.cat(
                [linear_resize(self.means[..., :self.bins], bins),
                 self.means[..., self.bins:]],
                -1)
            new_stds = torch.cat(
                [linear_resize(self.stds[..., :self.bins], bins),
                 self.stds[..., self.bins:]],
                -1)

            self.means = nn.parameter.Parameter(new_means, requires_grad=True)
            self.stds = nn.parameter.Parameter(new_stds, requires_grad=True)
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
    print(agent.controller.state_dict())
    actions = agent.act(8, enable_dropout=True)
    rewards = torch.rand(8)
    print(agent.act(1).size())
    agent.cache_batch(actions, rewards)
    agent.learn(n_steps=32)
    print(agent.controller.state_dict())
    policy = agent.decode_policy(actions[0], resize=False)

    xs = torch.rand(8, 3, 32, 32)
    for i in range(3):
        print(policy(xs).size())

    ct = BasicController(bag.n_ops, n_transforms=4, bins=3)
    print(ct(3).size(), ct.state_dict()['means'].size())
    ct.reset_bins(4)
    print(ct(3).size(), ct.state_dict()['means'].size())
    ct.reset_bins(5)
    print(ct(3).size(), ct.state_dict()['means'].size())

