"""
Python versions of functions for testing purposes etc.
"""
import numpy as np

def python_mvnpdf(data, means, covs):
    from pymc import mv_normal_cov_like as pdf_func

    results = []
    for i, datum in enumerate(data):
        for j, cov in enumerate(covs):
            mean = means[j]
            results.append(pdf_func(datum, mean, cov))

    return np.array(results).reshape((len(data), len(covs))).squeeze()

def python_sample_discrete(pmfs, draws=None):
    T, K = pmfs.shape
    output = np.empty(T, dtype=np.int32)
    if draws is None:
        draws = np.random.rand(T)

    # rescale
    pmfs = (pmfs.T / pmfs.sum(1)).T

    for i in xrange(T):
        the_sum = 0
        draw = draws[i]
        for j in xrange(K):
            the_sum += pmfs[i, j]

            if the_sum >= draw:
                output[i] = j
                break

    return output

if __name__ == '__main__':
    pmfs = np.random.randn(20, 5)
    pmfs = (pmfs.T - pmfs.min(1)).T


