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
import numpy as np
from pycuda.gpuarray import to_gpu
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

univ_setup_gpuarray = univ_setup + """
gpu_data = to_gpu(cpu_data)
"""

multivar_setup = """
# from __main__ import data, mean, std
import gpustats
import gpustats.util as util
import numpy as np
import testmod
from pycuda.gpuarray import to_gpu
import testmod
from numpy.linalg import cholesky as chol
import numpy.linalg as L


def next_multiple(k, p):
    if k.__mod__(p):
        return k + (p - k.__mod__(p))

    return k

PAD_MULTIPLE = 16
HALF_WARP = 16


def pad_data(data):
    n, k = data.shape

    if not k.__mod__(HALF_WARP):
        pad_dim = k + 1
    else:
        pad_dim = k

    if k != pad_dim:
        padded_data = np.empty((n, pad_dim), dtype=np.float32)
        padded_data[:, :k] = data

        return padded_data
    else:
        return prep_ndarray(data)

def prep_ndarray(arr):
    # is float32 and contiguous?
    if not arr.dtype == np.float32 or not arr.flags.contiguous:
        arr = np.array(arr, dtype=np.float32)

    return arr

def pack_params(means, chol_sigmas, logdets):
    to_pack = []
    for m, ch, ld in zip(means, chol_sigmas, logdets):
        to_pack.append(pack_pdf_params(m, ch, ld))

    return np.vstack(to_pack)

def pack_pdf_params(mean, chol_sigma, logdet):
    k = len(mean)
    mean_len = k
    chol_len = k * (k + 1) / 2
    mch_len = mean_len + chol_len

    packed_dim = next_multiple(mch_len + 2, PAD_MULTIPLE)

    packed_params = np.empty(packed_dim, dtype=np.float32)
    packed_params[:mean_len] = mean

    packed_params[mean_len:mch_len] = chol_sigma[np.tril_indices(k)]
    packed_params[mch_len:mch_len + 2] = 1, logdet

    return packed_params

k = %d

dim = 15
means = np.random.randn(k, dim)
covs = [util.random_cov(dim) for _ in xrange(k)]

cpu_data = np.random.randn(%d, dim)
gpu_data = cpu_data
"""

multivar_setup_gpuarray = multivar_setup + """
gpu_data = to_gpu(cpu_data)
"""

LOG_2_PI = np.log(2 * np.pi)

# def mvnpdf(data, mean, cov):
#     ichol_sigma = np.asarray(np.linalg.inv(np.linalg.cholesky(cov)))
#     # ichol_sigma = np.tril(ichol_sigma)
#     logdet = np.log(np.linalg.det(cov))
#     return [_mvnpdf(x, mean, ichol_sigma, logdet)
#             for x in data]

# def _mvnpdf(x, mean, ichol_sigma, logdet):
#     demeaned = x - mean
#     discrim = ((ichol_sigma * demeaned) ** 2).sum()
#     # discrim = np.dot(demeaned, np.dot(ichol_sigma, demeaned))
#     return - 0.5 * (discrim + logdet + LOG_2_PI * dim)

def get_timeit(stmt, setup, iter=10):
    return timeit.Timer(stmt, setup).timeit(number=iter) / iter

def compare_timings_single(n, setup=univ_setup):
    gpu = "gpustats.normpdf(gpu_data, mean, std, logged=False)"
    cpu = "norm.pdf(cpu_data, loc=mean, scale=std)"
    setup = setup % n
    return {'GPU' : get_timeit(gpu, setup, iter=1000),
            'CPU' : get_timeit(cpu, setup)}

def compare_timings_multi(n, setup=univ_setup):
    gpu = "gpustats.normpdf_multi(gpu_data, means, stds, logged=False)"
    cpu = """
for m, s in zip(means, stds):
    norm.pdf(cpu_data, loc=m, scale=s)
"""
    setup = setup % n
    return {'GPU' : get_timeit(gpu, setup, iter=100),
            'CPU' : get_timeit(cpu, setup)}


def mvcompare_timings(n, k=1, setup=multivar_setup):
    gpu = "gpustats.mvnpdf_multi(gpu_data, means, covs, logged=False)"
    cpu = """
ichol_sigmas = [L.inv(chol(sig)) for sig in covs]
logdets = [np.log(np.linalg.det(sig)) for sig in covs]
params = pack_params(means, covs, logdets)
testmod.cpu_mvnpdf(cpu_data, params, dim)
    """
    setup = setup % (k, n)
    return {'GPU' : get_timeit(gpu, setup, iter=100),
            'CPU' : get_timeit(cpu, setup)}

def get_timing_results(timing_f):
    lengths = [100, 1000, 10000, 100000, 1000000]

    result = {}
    for n in lengths:
        print n
        result[n] = timing_f(n)
    result = DataFrame(result).T
    result['Speedup'] = result['CPU'] / result['GPU']
    return result

# mvsingle = get_timing_results(mvcompare_timings)
# comp_gpu = lambda n: mvcompare_timings(n, setup=multivar_setup_gpuarray)
# mvsingle_gpu = get_timing_results(comp_gpu)
# multi_comp = lambda n: mvcompare_timings(n, k=16)
# mvmulti = get_timing_results(multi_comp)
# multi_comp_gpu = lambda n: mvcompare_timings(n, k=16,
#                                        setup=multivar_setup_gpuarray)
# mvmulti_gpu = get_timing_results(multi_comp_gpu)

single = get_timing_results(compare_timings_single)
comp_gpu = lambda n: compare_timings_single(n, setup=univ_setup_gpuarray)
single_gpu = get_timing_results(comp_gpu)
multi = get_timing_results(compare_timings_multi)
comp_gpu = lambda n: compare_timings_multi(n, setup=univ_setup_gpuarray)
multi_gpu = get_timing_results(comp_gpu)

data = DataFrame({
    'Single' : single['Speedup'],
    'Single (GPUArray)' : single_gpu['Speedup'],
    'Multi' : multi['Speedup'],
    'Multi (GPUArray)' : multi_gpu['Speedup'],
})


mvdata = DataFrame({
    'Single' : mvsingle['Speedup'],
    'Single (GPUArray)' : mvsingle_gpu['Speedup'],
    'Multi' : mvmulti['Speedup'],
    'Multi (GPUArray)' : mvmulti_gpu['Speedup'],
})

if __name__ == '__main__':
    import gpustats
    import numpy as np
    from scipy.stats import norm
    import testmod
    from numpy.linalg import cholesky as chol
    import numpy.linalg as L

    # dim = 15
    # k = 8
    # means = np.random.randn(k, dim)
    # covs = [np.asarray(util.random_cov(dim)) for _ in xrange(k)]

    # cpu_data = np.random.randn(100000, dim)
    # gpu_data = to_gpu(cpu_data)

    # ichol_sigmas = [L.inv(chol(sig)) for sig in covs]
    # logdets = [np.log(np.linalg.det(sig)) for sig in covs]
    # packed_params = pack_params(means, covs, logdets)

    # pdfs = gpustats.mvnpdf(cpu_data, means[0], covs[0])
    # pdfs = testmod.cpu_mvnpdf(cpu_data, packed_params, 15)

