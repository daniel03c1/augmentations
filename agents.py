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
                 lr=0.00035,
                 batch_size=1, 
                 epsilon=0.2,
                 online=True,
                 augmentation=None):
        self.device = get_default_device()
        self.net = net.to(self.device)
        self.name = name
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.memory = deque(maxlen=1024)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.online = online
        self.augmentation = augmentation

    def act(self, states):
        return self.net(states.to(self.device))

    def cache(self, state, prob, reward, next_state):
        self.memory.append((state.to(self.device), 
                            prob.to(self.device), 
                            reward.to(self.device), 
                            next_state.to(self.device)))

    def cache_batch(self, states, probs, rewards, next_states):
        for i in range(states.size(0)):
            self.cache(states[i], probs[i], rewards[i], next_states[i])

    def learn(self, n_steps=1):
        for i in range(n_steps):
            # inputs
            states, prior_probs, rewards, next_states = self.recall()
            if self.augmentation:
                states, prior_probs, rewards, next_states = self.augmentation(
                    states, prior_probs, rewards, next_states)
            prior_log_probs = prior_probs.detach().log()

            # train
            self.optimizer.zero_grad()

            probs = self.net(states)
            ratios = torch.exp(probs.log() - prior_log_probs)

            loss = -torch.min(
                ratios*rewards,
                torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon)*rewards)
            # entropy loss
            loss += 1e-5 * (probs * probs.log()).sum(-1).mean()

            torch.mean(loss).backward()
            self.optimizer.step()

        if self.online:
            self.clear_memory()

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        states, probs, rewards, next_states = map(torch.stack, zip(*batch))
        return states, probs, rewards, next_states

    def clear_memory(self):
        self.memory.clear()

    def save(self):
        torch.save(self.net.state_dict(), self.name)

