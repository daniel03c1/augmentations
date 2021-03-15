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
        self.np_img = np.array(self.pil_img).transpose(2, 0, 1) # to C, H, W
        self.torch_img = torch.Tensor(self.np_img) / 255.

    def test_autocontrast(self):
        target = PIL.ImageOps.autocontrast(self.pil_img)
        pred = AutoContrast(1.)(self.torch_img)
        self.compare(target, pred)

    def test_invert(self):
        target = PIL.ImageOps.invert(self.pil_img)
        pred = Invert(1.)(self.torch_img)
        self.compare(target, pred)

    def test_equalize(self):
        target = PIL.ImageOps.equalize(self.pil_img)
        pred = Equalize(1.)(self.torch_img)
        self.compare(target, pred)

    def test_solarize(self):
        mag = 110
        target = PIL.ImageOps.solarize(self.pil_img, mag)
        pred = Solarize(mag / 255.)(self.torch_img)
        self.compare(target, pred)

    def test_posterize(self):
        mag = 4
        target = PIL.ImageOps.posterize(self.pil_img, mag)
        pred = Solarize(1.)(self.torch_img)
        self.compare(target, pred)

    def test_sharpness(self):
        target = PIL.ImageEnhance.Sharpness(self.pil_img).enhance(1.3)
        pred = Solarize(0.3)(self.torch_img)
        self.compare(target, pred)

    def test_identity(self):
        target = self.pil_img
        pred = Identity(1.)(self.torch_img)
        self.compare(target, pred)

    def compare(self, pil_img, torch_img):
        self.assertTrue(np.allclose((np.array(pil_img)/255.).transpose(2, 0, 1),
                                    torch_img.numpy()))


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

