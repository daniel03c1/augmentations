import numpy as np
import os
import torch
import unittest
from agents import *


class AgentsTest(unittest.TestCase):
    def setUp(self):
        datasize, input_feat, n_actions = 32, 2, 3

        class Controller(nn.Module):
            ''' temporary class for tests '''
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(input_feat, n_actions)

            def forward(self, x):
                actions_dist = self.linear(x).softmax(-1)
                actions = torch.multinomial(actions_dist, 1)
                return actions, actions_dist

            def calculate_log_probs(self, actions, actions_dist):
                log_probs = (torch.eye(n_actions)[actions.squeeze(-1)] \
                             * actions_dist).sum(-1)
                return log_probs

            def calculate_entropy(self, actions_dist):
                return (actions_dist * actions_dist.log()).sum(-1)

        self.controller = Controller()

        self.states = torch.randn((datasize, input_feat))
        self.rewards = torch.randn(datasize)
        self.next_states = torch.randn((datasize, input_feat))

    def test_ppo_agent(self):
        ppo = PPOAgent(self.controller, 
                       'agents_test_ppo.pt', 
                       batch_size=32,
                       online=False)
        actions, dists = ppo.act(self.states)
        ppo.cache_batch(
            self.states, actions, dists, self.rewards, self.next_states)
        states, actions, dists, rewards, new_states = ppo.recall()

        base = ppo.learn(1).mean()
        test = ppo.learn(100).mean()
        self.assertGreater(base, test)
        ppo.clear_memory()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

