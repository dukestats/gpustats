from numpy.random import randn
from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as L
import scipy.special as sp
import pymc.flib as flib
import time
import testmod
import util
import pdb

def gen_testdata(n=100, k=4):
    # use static data to compare to R
    data = randn(n, k)
    mean = randn(k)

    np.savetxt('test_data', data)
    np.savetxt('test_mean', mean)

def load_testdata():
    data = np.loadtxt('test_data')
    mean = np.loadtxt('test_mean')
    cov = np.cov(data.T)


    return data, mean, cov

def bench(cpu_func, gpu_func, gruns=50):
    """

    """

    _s = time.clock()
    for i in xrange(gruns):
        testmod._mvnpdf(packed_data, packed_params, k).squeeze()

    gpu_speed = (time.clock() - _s) / gruns

    _s = time.clock()
    testmod.cpu_mvnpdf(packed_data, packed_params, k).squeeze()
    cpu_speed = (time.clock() - _s)
    print 'CPU speed: %.3f' % (cpu_speed * 1000)
    print 'GPU speed: %.3f' % (gpu_speed * 1000)
    print cpu_speed / gpu_speed

if __name__ == '__main__':
    k = 5
    n = 16

    testmod.set_device(0)

    data, mean, cov = load_testdata()

    j = 1

    n = 100
    k = 14

    data = randn(n, k)
    mean = randn(k)
    cov = np.array(util.random_cov(k), dtype=np.float32)

    j = 1

    packed_data = util.pack_data(data)

    chol_sigma = chol(cov)
    ichol_sigma = L.inv(chol_sigma)

    logdet = np.log(np.linalg.det(cov))

    means = (mean,) * j
    covs = (ichol_sigma,) * j
    logdets = (logdet,) * j

    packed_params = util.pack_params(means, covs, logdets)

    cpu_func = lambda: testmod.cpu_mvnpdf(packed_data, packed_params, k).squeeze()
    gpu_func = lambda: testmod._mvnpdf(packed_data, packed_params, k).squeeze()

    print cpu_func()
    print gpu_func()

    # bench(cpu_func, gpu_func, gruns=1)
