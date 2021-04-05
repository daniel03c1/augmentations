import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from utils import get_default_device


class PPOAgent:
    def __init__(self, 
                 net: nn.Module, 
                 name: str,
                 mem_maxlen: int,
                 lr=0.00035,
                 grad_norm=0.5,
                 batch_size=1, 
                 epsilon=0.2,
                 ent_coef=1e-5,
                 augmentation=None,
                 device=None):
        if device:
            self.device = device
        else:
            self.device = get_default_device()
        self.net = net.to(self.device)
        self.name = name
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.grad_norm = grad_norm

        self.memory = deque(maxlen=mem_maxlen)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.ent_coef = ent_coef
        self.augmentation = augmentation

    def act(self, *args, **kwargs):
        self.net.eval()
        return self.net(*args, **kwargs)

    def cache(self, state, action, dist, reward, next_state):
        self.memory.append((state.to(self.device), 
                            action.to(self.device).detach(),
                            dist.to(self.device).detach(), 
                            reward.to(self.device), 
                            next_state.to(self.device)))

    def cache_batch(self, states, actions, dists, rewards, next_states):
        for i in range(states.size(0)):
            self.cache(states[i], actions[i], dists[i], rewards[i], 
                       next_states[i])

    def learn(self, n_steps=1):
        self.net.train()

        for i in range(n_steps):
            # inputs
            memory = self.recall()
            if self.augmentation:
                memory = self.augmentation(*memory)
            states, actions, dists, rewards, next_states = memory

            self.optimizer.zero_grad()

            _, new_dists = self.net(states)

            # TODO: precalculate log_probs of dists
            ratios = self.net.calculate_log_probs(actions, new_dists) \
                   - self.net.calculate_log_probs(actions, dists).detach()
            # for stability
            ratios = torch.exp(torch.clamp(ratios, -5, 5))

            loss = -torch.min(
                ratios*rewards,
                torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon)*rewards)
            assert torch.isnan(loss).sum() == 0
            loss -= self.ent_coef * self.net.calculate_entropy(new_dists)
            assert torch.isnan(loss).sum() == 0

            torch.mean(loss).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 
                                           self.grad_norm)
            self.optimizer.step()

        return loss # the last loss

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, dists, rewards, next_states = map(
            torch.stack, zip(*batch))
        return states, actions, dists, rewards, next_states

    def clear_memory(self):
        self.memory.clear()

    def save(self, name=None):
        if not name:
            name = self.name
        if not name.endswith('.pt'):
            name += '.pt'
        torch.save(self.net.state_dict(), name)

    def decode_policy(self, *args, **kwargs):
        return self.net.decode_policy(*args, **kwargs)

    def apply_gamma(self, gamma):
        # gamma refers deprecation rate
        # it multiplies gamma to every reward in the queue
        for i in range(len(self.memory)):
            state, action, dist, reward, next_state = self.memory.popleft()
            reward *= gamma
            self.cache(state, action, dist, reward, next_state)

