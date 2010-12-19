cimport numpy as cnp
import numpy as np

cdef int is_contiguous(ndarray arr):
    return arr.flags.contiguous

cdef extern from "kernels.h":


    cudaError_t gpuMvNormalPDF(
        float* iData,
        float* iDensityInfo,
        float* oMeasure,
        int iD,
        int iN,
        int iTJ
        )

cdef struct Foo:
    cdef:
        int bar

def mvnpdf(ndarray data, ndarray mean, ndarray cov):
    output = np.empty_like(data)

    n, k = data.shape

    packed_params = np.empty(packed_dim, dtype=np.float32)
    gpuMvNormalPDF(data.data, packed.data, output.data,
                   k, n, 1)


