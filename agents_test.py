import numpy as np
import os
import torch
import unittest
from agents import *


class AgentsTest(unittest.TestCase):
    def setUp(self):
        datasize, input_feat = 32, 2
        output_feat = 1

        self.net = nn.Sequential(
            nn.Linear(input_feat, 5),
            nn.ReLU(),
            nn.Linear(5, output_feat))
        self.states = torch.rand((datasize, input_feat))
        self.rewards = torch.rand((datasize,))
        self.next_states = torch.rand((datasize, input_feat))

    def test_ppo_agent(self):
        ppo = PPOAgent(self.net, 'agents_test_ppo.pt')
        probs = ppo.act(self.states)
        ppo.cache_batch(self.states, probs, self.rewards, self.next_states)
        states, probs, rewards, new_states = ppo.recall()
        ppo.learn(10)
        ppo.clear_memory()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

