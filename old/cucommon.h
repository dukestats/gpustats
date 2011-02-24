/*
Common functions for GPUStats CUDA kernels and interface functions

 */
#ifndef __CUCOMMON_H__
#define __CUCOMMON_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int smem_size() {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  return deviceProp.sharedMemPerBlock;
}

int max_block_threads() {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  return deviceProp.maxThreadsPerBlock;
}

// Simple strided matrix data structure, far as I can tell there's little or no
// overhead in the compiled version.
typedef struct PMatrix {
  float* buf; // C-style row-major data
  int rows; // actual number of rows
  int cols; // actual number of columns
  int stride; // data length of row
} PMatrix;

void PMatrix_init(PMatrix* mat, float* data, int rows, int cols, int stride){
  mat->buf = data;
  mat->rows = rows;
  mat->cols = cols;
  mat->stride = stride;
}

typedef struct {
  int data_per_block;
  int params_per_block;
} BlockDesign;

int next_pow2(int k, int pow2) {
  // next highest power of two
  while (k <= pow2 / 2) pow2 /= 2;
  return pow2;
}

int get_boxes(int n, int box_size) {
  // how many boxes of size box_size are needed to hold n things
  return (n + box_size - 1) / box_size;
}

void inline h_to_d(float* h_ptr, float* d_ptr, size_t n){
  cudaError_t error;
  CATCH_ERR(cudaMemcpy(d_ptr, h_ptr, n * sizeof(float), cudaMemcpyHostToDevice));
}

void inline d_to_h(float* d_ptr, float* h_ptr, size_t n){
  cudaError_t error;
  CATCH_ERR(cudaMemcpy(h_ptr, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
}

#ifdef __cplusplus
}
#endif

#endif // __CUCOMMON_H__
