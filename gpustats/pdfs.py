from numpy.random import randn
from numpy.linalg import cholesky as chol
import numpy as np
import numpy.linalg as LA
import scipy.special as sp
import time

import gpustats.kernels as kernels
import gpustats.util as util
import gpustats.codegen as codegen
reload(kernels)

cu_module = codegen.get_full_cuda_module()

def mvnpdf(data, mean, cov):
    f = cu_module.get_function('pdf_mvnormal')

    padded_data = util.pad_data(data)

    ichol_sigma = LA.inv(chol(cov))
    logdet = np.log(LA.det(cov))
    padded_params = util.pack_params([mean], [ichol_sigma], [logdet])

    N = len(data)
    data_per, params_per = tune_blocksize(padded_data,
                                          padded_params)

    data_per = 256
    params_per = 1

    block_design = (data_per * params_per, 1, 1)

    design = np.array(((data_per, params_per) +
                       padded_data.shape +
                       (k,) +
                       padded_params.shape),
                      dtype=np.float32)

    dest = np.zeros((n, j), dtype=np.float32, order='F')

    grid_design =

    f(drv.Out(dest), drv.In(padded_data), drv.In(padded_params),
      drv.In(design),
      block=block_design, shared=14000)

def mvnpdf_multi(data, means, covs):
    f = cu_module.get_function('pdf_mvnormal')

    pass


def _pack_mvnpdf_params(means, ichol_sigmas, logdets):
    to_pack = []
    for m, ch, ld in zip(means, ichol_sigmas, logdets):
        to_pack.append(_pack_mvnpdf_params(m, ch, ld))

    return np.vstack(to_pack)

def _pack_mvnpdf_params(mean, ichol_sigma, logdet):
    k = len(mean)
    mean_len = k
    ichol_len = k * (k + 1) / 2
    mch_len = mean_len + ichol_len

    packed_dim = next_multiple(mch_len + 2, PAD_MULTIPLE)

    packed_params = np.empty(packed_dim, dtype=np.float32)
    packed_params[:mean_len] = mean

    packed_params[mean_len:mch_len] = ichol_sigma[np.tril_indices(k)]
    packed_params[mch_len:mch_len + 2] = 1, logdet

    return packed_params

def tune_blocksize(data, params, device=0):
    """
    For multivariate distributions-- what's the optimal block size given the
    gpu?

    Parameters
    ----------

    Returns
    -------
    """
    # TODO: how to figure out active device in this thread for the multigpu
    # case?
    pass

n = 256
k = 4

data = randn(n, k).astype(np.float32)
mean = randn(k)
cov = np.array(util.random_cov(k), dtype=np.float32)

