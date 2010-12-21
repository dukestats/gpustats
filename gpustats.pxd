cdef extern from "cuda.h":
    struct cudaError_t:
        pass
    char* cudaGetErrorString(cudaError_t err)

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

    cudaError_t gpuMvNormalPDF2(
        float* iData,
        float* iDensityInfo,
        float* oMeasure,
        int iD,
        int iN,
        int PACK_DIM,
        int DIM
        )

    void cpu_mvnormpdf(float* x, float* density, float * output, int D,
                       int N, int T)


