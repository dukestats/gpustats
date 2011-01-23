import nose
import unittest

from numpy.random import randn
from numpy.linalg import inv, cholesky as chol
import numpy as np
import pymc.distributions as pymc_dist

import util
import testmod

def python_mvnpdf(data, means, covs, k):
    actual_data = data[:, :k]

    pdf_func = pymc_dist.mv_normal_cov_like

    results = []
    for i, datum in enumerate(actual_data):
        for j, cov in enumerate(covs):
            mean = means[j]
            results.append(pdf_func(datum, mean, cov))

    return np.array(results).reshape((len(data), len(covs)))

def _make_test_case(n=1000, k=4, p=1):
    data = randn(n, k)
    covs = [util.random_cov(k) for _ in range(p)]
    means = (data.mean(0),) * p
    python_result = python_mvnpdf(data, means, covs, k)
    pdata, pparams = prep_inputs(data, means, covs)
    return pdata, pparams, python_result

def prep_inputs(data, means, covs):
    prepped_data = util.pad_data(data)
    ichols = [inv(chol(c)) for c in covs]
    logdets = [np.log(np.linalg.det(c)) for c in covs]
    prepped_params = util.pack_params(means, ichols, logdets)
    return prepped_data, prepped_params

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

    def _check_cpu_case(self, n, k, p):
        data, params, result = _make_test_case(n, k, p)
        cpu_result = testmod.cpu_mvnpdf(data, params, k)
        self.assert_((np.abs(cpu_result - result) < TOL).all())

    def _check_gpu_case(self, n, k, p):
        data, params, result = _make_test_case(n, k, p)
        gpu_result = testmod._mvnpdf(data, params, k)
        self.assert_((np.abs(gpu_result - result) < TOL).all())

    def test_cpu(self):
        for n, k, p in self.test_cases:
            self._check_cpu_case(n, k, p)

    def test_gpu(self):
        for n, k, p in self.test_cases:
            self._check_gpu_case(n, k, p)

if __name__ == '__main__':
    nose.runmodule(argv=['', '--pdb', '-v', '--pdb-failure'])
