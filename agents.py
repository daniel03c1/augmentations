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


class DiscretePPOAgent:
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

    def act(self, states, train=True, **kwargs):
        if train:
            self.net.train()
        else:
            self.net.eval()
        return self.net(states, **kwargs)

    def cache(self, state, action, reward, next_state):
        self.memory.append((state.to(self.device), 
                            action.to(self.device).detach(),
                            reward.to(self.device), 
                            next_state.to(self.device)))

    def cache_batch(self, states, actions, rewards, next_states):
        for i in range(states.size(0)):
            self.cache(states[i], actions[i], rewards[i], 
                       next_states[i])

    def learn(self, n_steps=1):
        self.net.train()

        for i in range(n_steps):
            self.optimizer.zero_grad()

            # inputs
            memory = self.recall()
            if self.augmentation:
                memory = self.augmentation(*memory)
            states, old_actions, rewards, next_states = memory

            actions = self.net(states)

            ratios = self.net.calculate_log_probs(actions) \
                   - self.net.calculate_log_probs(old_actions).detach()

            loss = -torch.min(
                ratios*rewards,
                ratios.clamp(1-self.epsilon, 1+self.epsilon)*rewards)
            loss -= self.ent_coef * self.net.calculate_entropy(actions)

            torch.mean(loss).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 
                                           self.grad_norm)
            self.optimizer.step()

        return loss # the last loss

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = map(
            torch.stack, zip(*batch))
        return states, actions, rewards, next_states

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
            state, action, reward, next_state = self.memory.popleft()
            reward *= gamma
            self.cache(state, action, reward, next_state)


class DiscretePPOAgentv2:
    def __init__(self, 
                 controller: nn.Module, 
                 valuator: nn.Module, 
                 name: str,
                 mem_maxlen: int,
                 batch_mem_maxlen: int,
                 lr=0.00035,
                 grad_norm=0.5,
                 batch_size=1, 
                 epsilon=0.2,
                 ent_coef=1e-5,
                 device=None,
                 **args):
        print(**args)
        if device:
            self.device = device
        else:
            self.device = get_default_device()
        self.controller = controller.to(self.device)
        self.valuator = valuator.to(self.device)

        self.name = name
        self.c_optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.v_optimizer = optim.Adam(self.valuator.parameters(), weight_decay=1e-5)
        self.v_criterion = nn.L1Loss()
        self.grad_norm = grad_norm

        self.sample_memory = deque(maxlen=mem_maxlen * batch_size)
        self.batch_memory = deque(maxlen=batch_mem_maxlen)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.ent_coef = ent_coef

    def act(self, states, train=True, **kwargs):
        if train:
            self.controller.train()
        else:
            self.controller.eval()
        return self.controller(states, **kwargs)

    def cache(self, state, action, reward, next_state):
        self.sample_memory.append((state.to(self.device), 
                                   action.to(self.device).detach(),
                                   reward.to(self.device), 
                                   next_state.to(self.device)))

    def cache_batch(self, states, actions, rewards, next_states):
        for i in range(states.size(0)):
            self.cache(states[i], actions[i], rewards[i], 
                       next_states[i])

        self.batch_memory.append((states.to(self.device),
                                  actions.to(self.device).detach(),
                                  rewards.to(self.device).detach(),
                                  next_states.to(self.device)))

    def learn(self, n_steps=1):
        # train valuator
        self.valuator.train()
        epochs = len(self.batch_memory)

        while True:
            total_loss = 0

            for i in range(epochs):
                # x: actions, y: rewards
                _, xs, ys, _ = random.sample(self.batch_memory, 1)[0]

                self.v_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.valuator(xs)[..., 0]
                    ys_hat = (ys_hat - ys_hat.mean()) / ys_hat.std().clamp(min=1e-8)
                    loss = self.v_criterion(ys_hat, ys)

                    loss.backward(retain_graph=True)
                    self.v_optimizer.step()

                    total_loss += (loss.detach() - total_loss) / (1+i)

            if total_loss < 1e-1:
                break
        self.valuator.eval()

        # train controller
        self.controller.train()

        for i in range(n_steps):
            self.c_optimizer.zero_grad()

            states, old_actions, rewards, _ = self.recall()
            rewards = self.valuator(old_actions)[..., 0]
            rewards = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-8)

            actions = self.controller(states)

            # PPO
            ratios = self.controller.calculate_log_probs(actions) \
                   - self.controller.calculate_log_probs(old_actions).detach()
            rewards = rewards.view(-1, *([1]*(len(ratios.shape)-1)))

            loss = -torch.min(
                ratios*rewards,
                ratios.clamp(1-self.epsilon, 1+self.epsilon)*rewards)

            # REINFORCE
            loss -= self.ent_coef * self.controller.calculate_entropy(actions)

            torch.mean(loss).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 
                                           self.grad_norm)
            self.c_optimizer.step()

        return loss # the last loss

    def recall(self):
        batch = random.sample(self.sample_memory, self.batch_size)
        states, actions, rewards, next_states = map(
            torch.stack, zip(*batch))
        return states, actions, rewards, next_states

    def clear_memory(self):
        self.sample_memory.clear()

    def save(self, name=None):
        if not name:
            name = self.name
        if not name.endswith('.pt'):
            name += '.pt'
        torch.save(self.controller.state_dict(), name)

    def decode_policy(self, *args, **kwargs):
        return self.controller.decode_policy(*args, **kwargs)

