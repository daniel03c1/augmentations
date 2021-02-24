import numpy as np
import os
import torch
import unittest
from agents import *
from models import PolicyController


class AgentsTest(unittest.TestCase):
    def setUp(self):
        class BOO: n_ops = 11 # temp class

        self.bag_of_ops = BOO()
        self.net = PolicyController(self.bag_of_ops)
        datasize, input_feat = 32, 2

        self.states = torch.zeros(datasize)
        self.rewards = torch.zeros(datasize)
        self.next_states = torch.zeros(datasize)

    def test_ppo_agent(self):
        ppo = PPOAgent(self.net, 'agents_test_ppo.pt')
        actions, dists = ppo.act(self.states)
        ppo.cache_batch(self.states, actions, dists, self.rewards, self.next_states)
        states, actions, dists, rewards, new_states = ppo.recall()
        ppo.learn(10)
        ppo.clear_memory()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

