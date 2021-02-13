import numpy as np
import os
import unittest
from torch import optim
from models import *


class ModelsTest(unittest.TestCase):
    def test_controller(self):
        output_size = 10
        n_subpolicies = 5
        batch_size = 4
        controller = Controller(output_size=output_size,
                                n_subpolicies=n_subpolicies)

        # test output shape
        x = torch.zeros((batch_size, 1), dtype=torch.long)
        policy = controller(x)
        self.assertEqual([*policy.size()], 
                         [batch_size, n_subpolicies*4, output_size])

        # test trainability
        opt = optim.SGD(controller.parameters(), lr=0.001, momentum=0.9)
        opt.zero_grad()
        policy.mean().backward()
        opt.step()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

