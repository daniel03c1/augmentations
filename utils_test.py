import numpy as np
import torch
import unittest
from utils import *


class UtilsTest(unittest.TestCase):
    def test_RunningStats(self):
        stats = RunningStats()

        # numpy test
        xs = np.random.randn(256)
        for x in xs:
            stats.push(x)
        self.assertAlmostEqual(xs.mean(), stats.mean())
        self.assertAlmostEqual(xs.std(), stats.std())

        # pytorch test
        stats.clear()
        xs = torch.rand(256, 11)
        for x in xs:
            stats.push(x)
        self.assertTrue(np.allclose(xs.mean(axis=0), stats.mean()))
        self.assertTrue(np.allclose(xs.std(axis=0), stats.std()))


if __name__ == '__main__':
    unittest.main()

