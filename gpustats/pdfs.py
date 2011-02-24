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
        f = cu_module.get_function('log_pdf_mvnormal')
    else:
        f = cu_module.get_function('pdf_mvnormal')

    ichol_sigmas = [LA.inv(chol(c)) for c in covs]
    logdets = [np.log(LA.det(c)) for c in covs]

    padded_data = util.pad_data(data)
    padded_params = _pack_mvnpdf_params(means, ichol_sigmas, logdets)

    ndata, nparams = len(data), len(covs)
    data_per, params_per = util.tune_blocksize(padded_data,
                                               padded_params)

    shared_mem = util.compute_shmem(padded_data, padded_params,
                                    data_per, params_per)

    block_design = (data_per * params_per, 1, 1)
    grid_design = (util.get_boxes(ndata, data_per),
                   util.get_boxes(nparams, params_per))

    design = np.array(((data_per, params_per) +
                       padded_data.shape +
                       (k,) +
                       padded_params.shape),
                      dtype=np.float32)

    dest = np.zeros((ndata, nparams), dtype=np.float32, order='F')

    f(drv.Out(dest), drv.In(padded_data), drv.In(padded_params),
      drv.In(design),
      block=block_design, grid=grid_design, shared=shared_mem)

    return dest

def python_mvnpdf(data, means, covs, k):
    import pymc.distributions as pymc_dist
    actual_data = data[:, :k]

    pdf_func = pymc_dist.mv_normal_cov_like

    results = []
    for i, datum in enumerate(actual_data):
        for j, cov in enumerate(covs):
            mean = means[j]
            results.append(pdf_func(datum, mean, cov))

    return np.array(results).reshape((len(data), len(covs)))

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
    n = 1e5
    k = 8

    np.random.seed(1)
    data = randn(n, k).astype(np.float32)
    mean = randn(k).astype(np.float32)
    cov = util.random_cov(k).astype(np.float32)

    result = mvnpdf(data, mean, cov)
    pyresult = python_mvnpdf(data, [mean], [cov], 8).squeeze()

    print result - pyresult
