import nose
import sys
import unittest

from numpy.random import randn
from numpy.linalg import inv, cholesky as chol
from numpy.testing import assert_almost_equal, assert_equal
import numpy as np

import scipy.stats as sp_stats

import gpustats as gps
import gpustats.compat as compat
import gpustats.util as util

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2

np.set_printoptions(suppress=True)

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
    result = gps.mvnpdf_multi(data, means, covs)

    return result, pyresult

def _compare_single(n, k):
    data, means, covs = _make_test_case(n, k, 1)

    mean = means[0]
    cov = covs[0]

    # cpu in PyMC
    pyresult = compat.python_mvnpdf(data, [mean], [cov]).squeeze()
    # gpu

    result = gps.mvnpdf(data, mean, cov)
    return result, pyresult

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
        a, b = _compare_multi(n, k, p)
        assert_almost_equal(a, b, DECIMAL_2)

    def _check_single(self, n, k):
        a, b = _compare_single(n, k)
        assert_almost_equal(a, b, DECIMAL_2)

    def test_multi(self):
        for n, k, p in self.test_cases:
            self._check_multi(n, k, p)

    def test_single(self):
        for n, k, p in self.test_cases:
            self._check_single(n, k)

class TestUnivariate(unittest.TestCase):
    def test_normal(self):
        test_cases = [
            (100, 0, 1),
            (100, .5, 2.5),
            (10, 5, 3),
            (2000, 1, 4)
        ]
        for n, mean, std in test_cases:
            data = randn(n)
            pyresult = sp_stats.norm.pdf(data, loc=mean, scale=std)

            result = gps.normpdf(data, mean, std, logged=True)
            assert_almost_equal(result, np.log(pyresult), DECIMAL_5)

    def test_normal_multi(self):
        means = np.random.randn(5)
        scales = np.ones(5)

        data = np.random.randn(10)
        result = gps.normpdf_multi(data, means, scales, logged=True)

        pyresult = np.empty_like(result)
        for i, (m, sc) in enumerate(zip(means, scales)):
            pyresult[:, i] = sp_stats.norm.pdf(data, loc=m, scale=sc)
        assert_almost_equal(result, np.log(pyresult), DECIMAL_5)

if __name__ == '__main__':
    # nose.runmodule(argv=['', '--pdb', '-v', '--pdb-failure'])
    pass
