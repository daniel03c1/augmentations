import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
import torch
import os
import unittest
from transforms import *


class TransformsTest(unittest.TestCase):
    def setUp(self):
        self.pil_img = PIL.Image.open('test_img.jpeg')
        self.torch_img = np.array(self.pil_img).transpose(2, 0, 1) # to C, H, W
        self.torch_img = torch.Tensor(self.torch_img)

    def test_autocontrast(self):
        target = PIL.ImageOps.autocontrast(self.pil_img)
        pred = AutoContrast(1.)(self.torch_img)

        self.assertEqual(np.array(pred/255).transpose(2, 0, 1),
                         pred.numpy())

    def test_invert(self):
        raise NotImplemented()

    def test_equalize(self):
        raise NotImplemented()

    def test_solarize(self):
        raise NotImplemented()

    def test_posterize(self):
        raise NotImplemented()

    def test_sharpness(self):
        raise NotImplemented()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

