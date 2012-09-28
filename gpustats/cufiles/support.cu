
#define LOG_2_PI 1.83787706640935f
#define LOG_PI 1.144729885849400f

__device__ int d_next_multiple(int k, int mult) {
  if (k % mult)
    return k + (mult - k % mult);
  else
    return k;
}

__device__ void copy_chunks(float* in_buf, float* out_buf,
                            unsigned int tid, unsigned int total) {
  for (unsigned int chunk = 0; chunk + tid < total; chunk += blockDim.x) {
    out_buf[chunk + tid] = in_buf[chunk + tid];
  }
}

__device__ void copy_chunks_strided(float* in_buf, float* out_buf,
                            unsigned int tid, unsigned int ncols, 
			    unsigned int nrows, unsigned int stride) {
  unsigned int outind = 0; unsigned int total = ncols*nrows;
  for (unsigned int chunk = 0; chunk + tid < total; chunk += blockDim.x) {
    outind = ((chunk + tid)/ncols)*stride + (chunk + tid) % ncols;
    out_buf[outind] = in_buf[chunk + tid];
  }
}


__device__ inline void atomic_add(float* address, float value){
#if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
  atomicAdd(address, value);
#elif __CUDA_ARCH__ >= 110
  float old = value;
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
#endif
}

