cimport numpy as cnp
from numpy cimport ndarray
import numpy as np

cimport gpustats as gps

import util

def set_device(device):
    '''
    Set the CUDA device
    '''
    gps.set_device(device)

def cpu_mvnpdf(ndarray packed_data, ndarray packed_params, int dim):
    n, j = len(packed_data), len(packed_params)

    cdef ndarray output = np.empty((n, j), dtype=np.float32)
    gps.cpu_mvnormpdf(<float*> packed_data.data,
                       <float*> packed_params.data,
                       <float*> output.data,
                       dim, n, j)

    return output

def mvnpdf(ndarray data, means, chol_sigmas, logdets):
    cdef ndarray output, packed_params, packed_data
    n, k = (<object> data).shape
    j = len(means)

    packed_params = util.pack_params(means, chol_sigmas, logdets)
    packed_data = util.pack_data(data)
    return _mvnpdf(packed_data, packed_params, k)

def _mvnpdf(ndarray packed_data, ndarray packed_params, int dim):
    n, k = (<object> packed_data).shape
    pn, pk = (<object> packed_params).shape
    cdef ndarray output = np.empty((n, pn), np.float32, order='F')
    gps.mvnpdf(<float*> packed_data.data,
                <float*> packed_params.data,
                <float*> output.data,
                dim, n, pn, pk, k)

    return output
