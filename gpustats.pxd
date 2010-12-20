cdef extern from "cuda.h":
    enum cudaError_t:
        pass

cdef extern from "mvnpdf.h":
    cudaError_t gpuMvNormalPDF(
        float* iData,
        float* iDensityInfo,
        float* oMeasure,
        int iD,
        int iN,
        int iTJ,
        int PACK_DIM,
        int DIM
        )
