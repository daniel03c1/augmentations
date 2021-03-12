import os
import unittest
import torchvision
from dataloader import *


class DataloaderTest(unittest.TestCase):
    def test_efficient_cifar10(self):
        dataset = EfficientCIFAR10('/media/data1/datasets/cifar', 
                                   train=False)

        # without transform
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual([*x.size()], [3, 32, 32])
        self.assertLessEqual(x.max(), 1)
        self.assertGreaterEqual(x.min(), 0)

        # with transform
        dataset.transform = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        new_x, new_y = dataset[0]
        self.assertIsInstance(new_x, torch.Tensor)
        self.assertEqual([*new_x.size()], [3, 32, 32])
        self.assertGreater((new_x - x).abs().sum().item(), 0)

    def test_efficient_cifar100(self):
        dataset = EfficientCIFAR100('/media/data1/datasets/cifar',
                                    train=False)

        # without transform
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual([*x.size()], [3, 32, 32])
        self.assertLessEqual(x.max(), 1)
        self.assertGreaterEqual(x.min(), 0)

        # with transform
        dataset.transform = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        new_x, new_y = dataset[0]
        self.assertIsInstance(new_x, torch.Tensor)
        self.assertEqual([*new_x.size()], [3, 32, 32])
        self.assertGreater((new_x - x).abs().sum().item(), 0)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

