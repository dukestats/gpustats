from numpy.random import randn
from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as LA

from pycuda.gpuarray import GPUArray, to_gpu
import gpustats.kernels as kernels
import gpustats.codegen as codegen
reload(codegen)
reload(kernels)
import gpustats.util as util
import pycuda.driver as drv

__all__ = ['mvnpdf', 'mvnpdf_multi', 'normpdf', 'normpdf_multi']

cu_module = codegen.get_full_cuda_module()

#-------------------------------------------------------------------------------
# Invokers for univariate and multivariate density functions conforming to the
# standard API

def _multivariate_pdf_call(cu_func, data, packed_params, get,
                           datadim=None):
    packed_params = util.prep_ndarray(packed_params)
    func_regs = cu_func.num_regs

    # Prep the data. Skip if gpudata ...
    if isinstance(data, GPUArray):
        padded_data = data
        if datadim==None:
            ndata, dim = data.shape
        else:
            ndata, dim = data.shape[0], datadim

    else:

        ndata, dim = data.shape
        padded_data = util.pad_data(data)


    nparams = len(packed_params)
    data_per, params_per = util.tune_blocksize(padded_data,
                                               packed_params,
                                               func_regs)

    blocksize = data_per * params_per

    shared_mem = util.compute_shmem(padded_data, packed_params,
                                    data_per, params_per)
    block_design = (data_per * params_per, 1, 1)
    grid_design = (util.get_boxes(ndata, data_per),
                   util.get_boxes(nparams, params_per))

    # see cufiles/mvcaller.cu
    design = np.array(((data_per, params_per) + # block design
                       padded_data.shape + # data spec
                       (dim,) + # non-padded number of data columns
                       packed_params.shape), # params spec
                      dtype=np.int32)

    if nparams == 1:
        gpu_dest = to_gpu(np.zeros(ndata, dtype=np.float32))
    else:
        gpu_dest = to_gpu(np.zeros((ndata, nparams),
                                   dtype=np.float32, order='C'))

    # Upload data if not already uploaded
    if not isinstance(padded_data, GPUArray):
        gpu_padded_data = to_gpu(padded_data)
    else:
        gpu_padded_data = padded_data

    gpu_packed_params = to_gpu(packed_params)

    params = (gpu_dest, gpu_padded_data, gpu_packed_params) + tuple(design)
    kwds = dict(block=block_design, grid=grid_design, shared=shared_mem)
    cu_func(*params, **kwds)

    if get:
        output = gpu_dest.get()
        if nparams > 1:
            output = output.reshape((nparams, ndata), order='C').T
        return output
    else:
        return gpu_dest

def _univariate_pdf_call(cu_func, data, packed_params, get):
    ndata = len(data)
    nparams = len(packed_params)

    func_regs = cu_func.num_regs

    packed_params = util.prep_ndarray(packed_params)

    data_per, params_per = util.tune_blocksize(data,
                                               packed_params,
                                               func_regs)

    shared_mem = util.compute_shmem(data, packed_params,
                                    data_per, params_per)

    block_design = (data_per * params_per, 1, 1)
    grid_design = (util.get_boxes(ndata, data_per),
                   util.get_boxes(nparams, params_per))

    # see cufiles/univcaller.cu

    gpu_dest = to_gpu(np.zeros((ndata, nparams), dtype=np.float32))
    gpu_data = data if isinstance(data, GPUArray) else to_gpu(data)
    gpu_packed_params = to_gpu(packed_params)

    design = np.array(((data_per, params_per) + # block design
                       (len(data),) +
                       packed_params.shape), # params spec
                      dtype=np.int32)

    cu_func(gpu_dest,
            gpu_data, gpu_packed_params, design[0],
            design[1], design[2], design[3], design[4],
            block=block_design, grid=grid_design, shared=shared_mem)

    if get:
        output = gpu_dest.get()
        if nparams > 1:
            output = output.reshape((nparams, ndata), order='C').T
        return output
    else:
        return gpu_dest

#-------------------------------------------------------------------------------
# Multivariate normal

def mvnpdf(data, mean, cov, weight=None, logged=True, get=True,
           datadim=None):
    """
    Multivariate normal density

    Parameters
    ----------

    Returns
    -------
    """
    return mvnpdf_multi(data, [mean], [cov],
                        logged=logged, get=get,
                        datadim=datadim).squeeze()

def mvnpdf_multi(data, means, covs, weights=None, logged=True,
                 get=True, datadim=None):
    """
    Multivariate normal density with multiple sets of parameters

    Parameters
    ----------
    data : ndarray (n x k)
    covs : sequence of 2d k x k matrices (length j)
    weights : ndarray (length j)
        Multiplier for component j, usually will sum to 1

    get = True leaves the result on the GPU
    without copying back.

    If data has already been padded, the orginal dimension
    must be passed in datadim

    It data is of GPUarray type, the data is assumed to be
    padded, and datadim will need to be passed if padding
    was needed.

    Returns
    -------
    densities : n x j
    """
    if logged:
        cu_func = cu_module.get_function('log_pdf_mvnormal')
    else:
        cu_func = cu_module.get_function('pdf_mvnormal')

    assert(len(covs) == len(means))

    ichol_sigmas = [LA.inv(chol(c)) for c in covs]
    logdets = [np.log(LA.det(c)) for c in covs]

    if weights is None:
        weights = np.ones(len(means))

    packed_params = _pack_mvnpdf_params(means, ichol_sigmas, logdets, weights)

    return _multivariate_pdf_call(cu_func, data, packed_params,
                                  get, datadim)

def _pack_mvnpdf_params(means, ichol_sigmas, logdets, weights):
    to_pack = []
    for m, ch, ld, w in zip(means, ichol_sigmas, logdets, weights):
        to_pack.append(_pack_mvnpdf_params_single(m, ch, ld, w))

    return np.vstack(to_pack)

def _pack_mvnpdf_params_single(mean, ichol_sigma, logdet, weight=1):
    PAD_MULTIPLE = 16
    k = len(mean)
    mean_len = k
    ichol_len = k * (k + 1) / 2
    mch_len = mean_len + ichol_len

    packed_dim = util.next_multiple(mch_len + 2, PAD_MULTIPLE)

    packed_params = np.empty(packed_dim, dtype=np.float32)
    packed_params[:mean_len] = mean

    packed_params[mean_len:mch_len] = ichol_sigma[np.tril_indices(k)]
    packed_params[mch_len:mch_len + 2] = weight, logdet

    return packed_params

#-------------------------------------------------------------------------------
# Univariate normal

def normpdf(x, mean, std, logged=True, get=True):
    """
    Normal (Gaussian) density

    Parameters
    ----------

    Returns
    -------
    """
    return normpdf_multi(x, [mean], [std], logged=logged, get=get).squeeze()

def normpdf_multi(x, means, std, logged=True, get=True):
    if logged:
        cu_func = cu_module.get_function('log_pdf_normal')
    else:
        cu_func = cu_module.get_function('pdf_normal')

    packed_params = np.c_[means, std]

    if not isinstance(x, GPUArray):
        x = util.prep_ndarray(x)

    return _univariate_pdf_call(cu_func, x, packed_params, get)

if __name__ == '__main__':
    import gpustats.compat as compat

    n = 1e5
    k = 8

    np.random.seed(1)
    data = randn(n, k).astype(np.float32)
    mean = randn(k).astype(np.float32)
    cov = util.random_cov(k).astype(np.float32)

    result = mvnpdf_multi(data, [mean, mean], [cov, cov])
    # pyresult = compat.python_mvnpdf(data, [mean], [cov]).squeeze()
    # print result - pyresult
