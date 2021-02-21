import os
import unittest
import torch
from wideresnet import *


class WideResNetTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.n_classes = 10
        self.x = torch.rand(self.batch_size, 3, 32, 32)

    def test_wide_resnet(self):
        model = WideResNet(28, 10, 0.3, n_classes=self.n_classes)
        self.assertEqual([*model(self.x).size()],
                         [self.batch_size, self.n_classes])

    def test_wide_resnet_block(self):
        block = WideResNetBlock(3, 16, 0.3, 1)
        self.assertEqual([*block(self.x).size()],
                         [self.batch_size, 16, 32, 32])

        block = WideResNetBlock(3, 12, 0.3, 2)
        self.assertEqual([*block(self.x).size()],
                         [self.batch_size, 12, 16, 16])


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

