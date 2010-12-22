cimport numpy as cnp
from numpy cimport ndarray
import numpy as np

cimport gpustats as gps

cdef int is_contiguous(ndarray arr):
    return arr.flags.contiguous

def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result

def rand_cov(k):
    correls = 2 * np.random.rand(k * (k + 1) / 2) - 1
    corrmat = unvech(correls)

    vols = np.random.randn(k)

    return corrmat * np.outer(vols, vols)

def prep_ndarray(ndarray arr):
    # is float32 and contiguous?
    if not arr.dtype == np.float32 or not is_contiguous(arr):
        arr = np.array(arr, dtype=np.float32)

    return arr

cdef int PAD_MULTIPLE = 16

cdef int DATA_PAD_MULTIPLE = 8

def next_multiple(k, p):
    if k % p:
        return k + (p - k % p)

    return k

def pack_data(data):
    n, k = data.shape

    pad_dim = next_multiple(k, DATA_PAD_MULTIPLE)

    if k != pad_dim:
        padded_data = np.empty((n, pad_dim), dtype=np.float32)
        padded_data[:, :k] = data

        return padded_data
    else:
        return prep_ndarray(data)

def mvnpdf(ndarray data, means, chol_sigmas, logdets):
    cdef ndarray output, packed_params, packed_data
    cdef gps.cudaError_t res
    n, k = (<object> data).shape
    j = len(means)

    packed_params = pack_params(means, chol_sigmas, logdets)
    packed_data = pack_data(data)

    output = np.empty((n, j), np.float32)

    gps.mvnpdf2(<float*> packed_data.data,
                 <float*> packed_params.data,
                 <float*> output.data,
                 k, n, j,
                 packed_params.shape[1],
                 packed_data.shape[1])

    return output

def mvnpdf2(ndarray data, means, chol_sigmas, logdets):
    cdef ndarray output, packed_params, packed_data
    cdef gps.cudaError_t res
    n, k = (<object> data).shape

    packed_params = pack_params(means, chol_sigmas, logdets)
    packed_data = pack_data(data)

    j = len(means)
    output = np.empty((n, j), dtype=np.float32)

    gps.gpuMvNormalPDF(<float*> packed_data.data,
                        <float*> packed_params.data,
                        <float*> output.data,
                        k, n, j,
                        packed_params.shape[1],
                        packed_data.shape[1])

    return output


def pack_params(means, chol_sigmas, logdets):
    to_pack = []
    for m, ch, ld in zip(means, chol_sigmas, logdets):
        to_pack.append(pack_pdf_params(m, ch, ld))

    return np.vstack(to_pack)

def pack_pdf_params(ndarray mean, ndarray chol_sigma, logdet):
    '''


    '''
    cdef int k, packed_dim

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

def cpu_mvnpdf(ndarray data, means, chol_sigmas, logdets):
    cdef ndarray output, packed_params, packed_data
    n, k = (<object> data).shape

    packed_data = pack_data(data)
    packed_params = pack_params(means, chol_sigmas, logdets)
    j = len(means)

    output = np.empty((n, j), dtype=np.float32)

    gps.cpu_mvnormpdf(<float*> packed_data.data,
                       <float*> packed_params.data,
                       <float*> output.data,
                       k, n, j)

    return output
