from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as L

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

