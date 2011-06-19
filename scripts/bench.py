from pandas import *

import numpy as np

from pycuda.gpuarray import to_gpu
import gpustats
import gpustats.util as util
from scipy.stats import norm
import timeit

data = np.random.randn(1000000)
mean = 20
std = 5

univ_setup = """
# from __main__ import data, mean, std
import numpy as np
k = 8
means = np.random.randn(k)
stds = np.abs(np.random.randn(k))

mean = 20
std = 5
import gpustats
from scipy.stats import norm
cpu_data = np.random.randn(%d)
gpu_data = cpu_data
"""

univ_setup_gpuarray = """
# from __main__ import data, mean, std
import numpy as np
import gpustats
from scipy.stats import norm
from pycuda.gpuarray import to_gpu
k = 8
means = np.random.randn(k)
stds = np.abs(np.random.randn(k))

mean = 20
std = 5
cpu_data = np.random.randn(%d)
gpu_data = to_gpu(cpu_data)
"""

multivar_setup = """
# from __main__ import data, mean, std
import gpustats
import gpustats.util as util
import numpy as np
from scipy.stats import norm
dim = 15
k = 8
means = np.random.randn(k, dim)
covs = [util.random_cov(dim) for _ in xrange(k)]

cpu_data = np.random.randn(%d, dim)
gpu_data = cpu_data
"""
LOG_2_PI = np.log(2 * np.pi)

def mvnpdf(data, mean, cov):
    ichol_sigma = np.asarray(np.linalg.inv(np.linalg.cholesky(cov)))
    # ichol_sigma = np.tril(ichol_sigma)
    logdet = np.log(np.linalg.det(cov))
    return [_mvnpdf(x, mean, ichol_sigma, logdet)
            for x in data]

def _mvnpdf(x, mean, ichol_sigma, logdet):
    demeaned = x - mean
    discrim = ((ichol_sigma * demeaned) ** 2).sum()
    # discrim = np.dot(demeaned, np.dot(ichol_sigma, demeaned))
    return - 0.5 * (discrim + logdet + LOG_2_PI * dim)

def get_timeit(stmt, setup, n, iter=50):
    setup = setup % n
    return timeit.Timer(stmt, setup).timeit(number=iter) / iter

def compare_timings_single(n, setup=univ_setup):
    gpu = "gpustats.normpdf(gpu_data, mean, std, logged=False)"
    cpu = "norm.pdf(cpu_data, loc=mean, scale=std)"
    return {'GPU' : get_timeit(gpu, setup, n),
            'CPU' : get_timeit(cpu, setup, n)}

def mvcompare_timings_single(n, setup=univ_setup):
    gpu = "gpustats.normpdf(gpu_data, mean, std, logged=False)"
    cpu = "norm.pdf(cpu_data, loc=mean, scale=std)"
    return {'GPU' : get_timeit(gpu, setup, n),
            'CPU' : get_timeit(cpu, setup, n)}

def compare_timings_multi(n, setup=univ_setup):
    gpu = "gpustats.normpdf_multi(gpu_data, means, stds, logged=False)"
    cpu = """
for m, s in zip(means, stds):
    norm.pdf(cpu_data, loc=m, scale=s)
"""
    return {'GPU' : get_timeit(gpu, univ_setup, n),
            'CPU' : get_timeit(cpu, univ_setup, n)}

def get_timing_results(timing_f):
    lengths = [100, 1000, 10000, 100000, 1000000]

    result = {}
    for n in lengths:
        result[n] = timing_f(n)
    result = DataFrame(result).T
    result['Speedup'] = result['CPU'] / result['GPU']
    return result

# single = get_timing_results(compare_timings_single)
# comp_gpu = lambda n: compare_timings_single(n, setup=univ_setup_gpuarray)
# single_gpu = get_timing_results(comp_gpu)
# multi = get_timing_results(compare_timings_multi)
# comp_gpu = lambda n: compare_timings_multi(n, setup=univ_setup_gpuarray)
# multi_gpu = get_timing_results(compare_timings_multi)

if __name__ == '__main__':
    import gpustats
    import gpustats.util as util
    import numpy as np
    from scipy.stats import norm
    dim = 15
    k = 8
    means = np.random.randn(k, dim)
    covs = [util.random_cov(dim) for _ in xrange(k)]

    cpu_data = np.random.randn(100000, dim)
    gpu_data = to_gpu(cpu_data)

    pdfs = gpustats.mvnpdf(cpu_data, means[0], covs[0])

