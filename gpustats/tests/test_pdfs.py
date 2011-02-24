import nose
import sys
import unittest

from numpy.random import randn
from numpy.linalg import inv, cholesky as chol
import numpy as np

import gpustats.pdfs as pdfs
import gpustats.compat as compat
import gpustats.util as util

from pandas.util.testing import debug

def _make_test_case(n=1000, k=4, p=1):
    data = randn(n, k)
    covs = [util.random_cov(k) for _ in range(p)]
    means = [randn(k) for _ in range(p)]
    return data, means, covs

# debugging...

def _compare_multi(n, k, p):
    data, means, covs = _make_test_case(n, k, p)

    # cpu in PyMC
    pyresult = compat.python_mvnpdf(data, means, covs)

    # gpu
    result = pdfs.mvnpdf_multi(data, means, covs)

    diff = result - pyresult

    return diff

def _compare_single(n, k):
    data, means, covs = _make_test_case(n, k, 1)

    mean = means[0]
    cov = covs[0]

    # cpu in PyMC
    pyresult = compat.python_mvnpdf(data, [mean], [cov]).squeeze()
    # gpu
    result = pdfs.mvnpdf(data, mean, cov)
    diff = result - pyresult

    return diff

# get some 1e-4 differences from FP error...
TOL = 1e-3

class TestMVN(unittest.TestCase):
    # ndata, dim, ncomponents
    test_cases = [(1000, 4, 1),
                  (1000, 4, 16),
                  (1000, 4, 32),
                  (1000, 4, 64),
                  (1000, 7, 64),
                  (1000, 8, 64),
                  (1000, 14, 32),
                  (1000, 16, 32),
                  (250, 25, 32),
                  (10, 15, 2)]

    def _check_multi(self, n, k, p):
        diff = _compare_multi(n, k, p)
        self.assert_((np.abs(diff) < TOL).all())

    def _check_single(self, n, k):
        diff = _compare_single(n, k)
        self.assert_((np.abs(diff) < TOL).all())

    def test_multi(self):
        for n, k, p in self.test_cases:
            self._check_multi(n, k, p)

    def test_single(self):
        for n, k, p in self.test_cases:
            self._check_single(n, k)

if __name__ == '__main__':
    # nose.runmodule(argv=['', '--pdb', '-v', '--pdb-failure'])
    pass
