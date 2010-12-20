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

def mvnpdf(ndarray data, ndarray mean, ndarray cov):
    cdef ndarray output, packed_params
    n, k = (<object> data).shape

    output = np.empty_like(data)
    packed_dim = np.empty(k * (k + 3) / 2)

    packed_params = np.empty(packed_dim, dtype=np.float32)
    gps.gpuMvNormalPDF(<float*> data.data,
                        <float*> packed_params.data,
                        <float*> output.data,
                        k, n, 1, 16, 16)


