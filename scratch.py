from numpy.random import randn
from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as L
import scipy.special as sp
import pymc.flib as flib
import time

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

    # n = 1e6
    # k = 10

    # data = randn(n, k)
    # mean = randn(k)
    # cov = random_cov(k) # np.cov(data.T)

    chol_sigma = chol(cov)
    ichol_sigma = L.inv(chol_sigma)

    logdet = np.log(np.linalg.det(cov))

    print flib.chol_mvnorm(data[0], mean, chol_sigma)

    import testmod
    print testmod.cpu_mvnpdf(data, mean, ichol_sigma, logdet)
    print testmod.mvnpdf(data, mean, ichol_sigma, logdet)

    # runs = 10

    # _s = time.clock()
    # for i in xrange(runs):
    #     print '.'
    #     testmod.mvnpdf(data, mean, ichol_sigma, logdet)

    # gpu_speed = (time.clock() - _s) / runs

    # _s = time.clock()
    # for i in xrange(runs):
    #     print '.'
    #     testmod.cpu_mvnpdf(data, mean, ichol_sigma, logdet)

    # print ''

    # cpu_speed = (time.clock() - _s) / runs

    # print 'CPU speed: %.3f' % (cpu_speed * 1000)
    # print 'GPU speed: %.3f' % (gpu_speed * 1000)




