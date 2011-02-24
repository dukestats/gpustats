from numpy.random import randn
from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as LA

import gpustats.kernels as kernels
import gpustats.codegen as codegen
reload(codegen)
reload(kernels)
import gpustats.util as util
import pycuda.driver as drv

from pandas.util.testing import set_trace as st

cu_module = codegen.get_full_cuda_module()

def mvnpdf(data, mean, cov, logged=True):
    """
    Multivariate normal density

    Parameters
    ----------

    Returns
    -------
    """
    return mvnpdf_multi(data, [mean], [cov]).squeeze()

def mvnpdf_multi(data, means, covs, logged=True):
    """
    Multivariate normal density with multiple sets of parameters

    Parameters
    ----------
    data : ndarray (n x k)
    covs : sequence of 2d k x k matrices (length j)
    weights : ndarray (length j)

    Returns
    -------
    densities : n x j
    """
    if logged:
        cuda_func = cu_module.get_function('log_pdf_mvnormal')
    else:
        cuda_func = cu_module.get_function('pdf_mvnormal')

    ichol_sigmas = [LA.inv(chol(c)) for c in covs]
    logdets = [np.log(LA.det(c)) for c in covs]

    padded_data = util.pad_data(data)
    padded_params = _pack_mvnpdf_params(means, ichol_sigmas, logdets)

    ndata, dim = data.shape

    nparams = len(covs)
    data_per, params_per = util.tune_blocksize(padded_data,
                                               padded_params)

    shared_mem = util.compute_shmem(padded_data, padded_params,
                                    data_per, params_per)

    block_design = (data_per * params_per, 1, 1)
    grid_design = (util.get_boxes(ndata, data_per),
                   util.get_boxes(nparams, params_per))

    design = np.array(((data_per, params_per) + # block design
                       padded_data.shape + # data spec
                       (dim,) + # non-padded number of data columns
                       padded_params.shape), # params spec
                      dtype=np.float32)

    dest = np.zeros((ndata, nparams), dtype=np.float32, order='F')

    cuda_func(drv.Out(dest),
              drv.In(padded_data), drv.In(padded_params), drv.In(design),
              block=block_design, grid=grid_design, shared=shared_mem)

    return dest

def _pack_mvnpdf_params(means, ichol_sigmas, logdets):
    to_pack = []
    for m, ch, ld in zip(means, ichol_sigmas, logdets):
        to_pack.append(_pack_mvnpdf_params_single(m, ch, ld))

    return np.vstack(to_pack)

def _pack_mvnpdf_params_single(mean, ichol_sigma, logdet):
    PAD_MULTIPLE = 16
    k = len(mean)
    mean_len = k
    ichol_len = k * (k + 1) / 2
    mch_len = mean_len + ichol_len

    packed_dim = util.next_multiple(mch_len + 2, PAD_MULTIPLE)

    packed_params = np.empty(packed_dim, dtype=np.float32)
    packed_params[:mean_len] = mean

    packed_params[mean_len:mch_len] = ichol_sigma[np.tril_indices(k)]
    packed_params[mch_len:mch_len + 2] = 1, logdet

    return packed_params

if __name__ == '__main__':
    import gpustats.compat as compat

    n = 1e5
    k = 8

    np.random.seed(1)
    data = randn(n, k).astype(np.float32)
    mean = randn(k).astype(np.float32)
    cov = util.random_cov(k).astype(np.float32)

    result = mvnpdf(data, mean, cov)
    pyresult = compat.python_mvnpdf(data, [mean], [cov], 8).squeeze()

    print result - pyresult
