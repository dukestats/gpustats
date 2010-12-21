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

    void mvnpdf2(float* h_data,
                 float* h_params,
                 float* h_pdf,
                 int data_dim,
                 int total_obs,
                 int param_stride,
                 int data_stride)

    void cpu_mvnormpdf(float* x, float* density, float * output, int D,
                       int N, int T)


