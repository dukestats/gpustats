cdef extern from "cuda.h":
    struct cudaError_t:
        pass
    char* cudaGetErrorString(cudaError_t err)

cdef extern from "common.h":
    struct PMatrix:
        float* data
        int rows
        int cols
        int stride

    void PMatrix_init(float* d, int r, int c, int s)

    void set_device(int device)

cdef extern from "mvnpdf.h":
    void mvnpdf(float* h_data,
                float* h_params,
                float* h_pdf,
                int data_dim,
                int total_obs,
                int nparams,
                int param_stride,
                int data_stride) nogil

    void cpu_mvnpdf(float* x, float* density, float * output, int D,
                    int padded_dim, int N, int T) nogil


