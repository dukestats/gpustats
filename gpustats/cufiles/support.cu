
#define LOG_2_PI 1.83787706640935
#define LOG_PI 1.144729885849400

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


// bar
