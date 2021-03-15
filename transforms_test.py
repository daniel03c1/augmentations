import numpy as np
import os
import unittest
from transforms import *


class TransformsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_autocontrast(self):
        raise NotImplemented()

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

