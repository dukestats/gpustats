from numpy.random import randn
from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as L
import scipy.special as sp
import pymc.flib as flib
import time
import testmod

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

def random_cov(k):
    ch = np.zeros((k, k), dtype=np.float32)

    ch[np.tril_indices(k)] = randn(k * (k + 1) / 2)
    cov = np.dot(ch, ch.T)
    flib.symmetrize(cov)

    return cov

if __name__ == '__main__':
    k = 5
    n = 16

    data, mean, cov = load_testdata()

    j = 1

    n = 1e4
    k = 16

    data = randn(n, k)
    mean = randn(k)
    cov = random_cov(k) # np.cov(data.T)

    j = 256

    packed_data = testmod.pack_data(data)

    chol_sigma = chol(cov)
    ichol_sigma = L.inv(chol_sigma)

    logdet = np.log(np.linalg.det(cov))

    means = (mean,) * j
    covs = (ichol_sigma,) * j
    logdets = (logdet,) * j

    print flib.chol_mvnorm(data[0], mean, chol_sigma)

    packed_params = testmod.pack_params(means, covs, logdets)

    r1 = testmod.cpu_mvnpdf(packed_data, packed_params, k).squeeze()
    r2 = testmod._mvnpdf(packed_data, packed_params, k).squeeze()

    print r1
    print r2
    diff = np.where(np.abs(r1 - r2) < 1e-4, 0, r1 - r2)
    print diff.sum() / np.prod(diff.shape)

    print diff[diff.sum(1) != 0], np.arange(len(diff))[diff.sum(1) != 0]

    print r2[0][:32]
    print packed_data[0]

    # gruns = 50

    # _s = time.clock()
    # for i in xrange(gruns):
    #     testmod._mvnpdf(packed_data, packed_params, k).squeeze()

    # gpu_speed = (time.clock() - _s) / gruns

    # print 'done with gpu'

    # cruns = 1
    # _s = time.clock()
    # for i in xrange(cruns):
    #     testmod.cpu_mvnpdf(packed_data, packed_params, k).squeeze()

    # cpu_speed = (time.clock() - _s) / cruns
    # print 'done with cpu'

    # print 'CPU speed: %.3f' % (cpu_speed * 1000)
    # print 'GPU speed: %.3f' % (gpu_speed * 1000)
    # print cpu_speed / gpu_speed
