from numpy.random import randn
from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as L
import scipy.special as sp
import pymc.flib as flib
import time
import testmod

# def test(k=4, n=100):
def gen_testdata():
    # use static data to compare to R
    data = randn(n, k)
    mean = randn(k)

def load_testdata():
    data = np.loadtxt('test_data')
    mean = np.loadtxt('test_mean')
    cov = np.cov(data.T)

    return data, mean, cov

def logpdf(x, mean, sig):
    log2pi = np.log(2 * np.pi)
    logdet = np.log(L.det(sig))

    print logdet

    cons = 0.5 * (len(x) * log2pi + logdet)

    xdm = x - mean

    kernel = 0.5 * np.dot(xdm, np.dot(L.inv(sig), xdm))
    print kernel

    return - (cons + kernel)

def logpdf2(x, mean, sig):
    log2pi = np.log(2 * np.pi)
    logdet = np.log(L.det(sig))

    cons = 0.5 * (len(x) * log2pi + logdet)

    xdm = x - mean

    ch = chol(sig)

    xch = L.solve(ch, xdm)

    kernel = 0.5 * np.dot(xdm, xch)

    print kernel

    return - (cons + kernel)

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

    j = 32

    n = 1e5
    k = 14

    data = randn(n, k)
    mean = randn(k)
    cov = random_cov(k) # np.cov(data.T)

    j = 128

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
    r2 = testmod.mvnpdf3(packed_data, packed_params, k).squeeze()

    print r1
    print r2
    diff = np.where(np.abs(r1 - r2) < 1e-4, 0, r1 - r2)
    print diff.sum() / np.prod(diff.shape)

    # print diff[diff.sum(1) != 0], np.arange(len(diff))[diff.sum(1) != 0]

    print r2[0][:32]
    print packed_data[0]

    gruns = 1

    _s = time.clock()
    for i in xrange(gruns):
        testmod.mvnpdf3(packed_data, packed_params, k).squeeze()

    gpu_speed = (time.clock() - _s) / gruns

    cruns = 1
    _s = time.clock()
    for i in xrange(cruns):
        testmod.cpu_mvnpdf(packed_data, packed_params, k).squeeze()

    cpu_speed = (time.clock() - _s) / cruns

    print 'CPU speed: %.3f' % (cpu_speed * 1000)
    print 'GPU speed: %.3f' % (gpu_speed * 1000)
    print cpu_speed / gpu_speed
