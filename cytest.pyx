cimport numpy as cnp
from numpy cimport ndarray
import numpy as np

cimport gpustats as gps

def set_device(device):
    '''
    Set the CUDA device
    '''
    gps.set_device(device)

def cpu_mvnpdf(ndarray packed_data, ndarray packed_params, int dim):
    n, j = len(packed_data), len(packed_params)

    padded_dim = (<object> packed_data).shape[1]

    cdef ndarray output = np.empty((n, j), dtype=np.float32)
    gps.cpu_mvnpdf(<float*> packed_data.data,
                    <float*> packed_params.data,
                    <float*> output.data,
                    dim, padded_dim, n, j)

    return output

def mvn_call(ndarray packed_data, ndarray packed_params, int dim):
    '''
    Invoke MVN kernel on prepared data

    Releases GIL
    '''
    cdef int n, k, pn, pk

    n, k = (<object> packed_data).shape
    pn, pk = (<object> packed_params).shape

    cdef ndarray output = np.empty((n, pn), np.float32, order='F')

    with nogil:
        gps.mvnpdf(<float*> packed_data.data,
                    <float*> packed_params.data,
                    <float*> output.data,
                   dim, n, pn, pk, k)

    return output
