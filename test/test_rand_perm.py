import unittest

import torch

from src.data.PermutedMNIST import RandomPermutation


class MyTestCase(unittest.TestCase):
    def test_RandomPermutation(self):
        n = 10
        in_img = torch.rand(3, 224, 224)
        for i in range(n):
            rand_perm = RandomPermutation(0, 0, 224, 224)
            permuted = rand_perm(in_img.clone())
            for j in range(n):
                self.assertTrue(torch.equal(permuted, rand_perm(in_img.clone())))


if __name__ == '__main__':
    unittest.main()
