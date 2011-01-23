/*
  Multivariate normal pdf implementation
 */

#ifndef _INCLUDED_MVNPDF
#define _INCLUDED_MVNPDF

#ifdef __cplusplus
extern "C" {
#endif

#include "mvnpdf.h"
#include "cucommon.h"

#define DEBUG

int compute_shmem(PMatrix* data, PMatrix* params, int nparams, int ndata) {
  // to hold specified about of data, parameters, and results
  int result_space = nparams * ndata;
  int param_space = params->stride * nparams;
  int data_space = data->stride * ndata;

  return sizeof(float) * (result_space + param_space + data_space);
}

// Compute "optimal" block size given number of data points / parameters
void get_tuned_layout(BlockDesign* info, PMatrix* data, PMatrix* params,
                      int max_block_params) {
  // query the device for smem / max # of threads
  int max_smem = smem_size() / 10 * 9;
  int max_threads = max_block_threads();

  // at most max_block_params sets of density parameters per block
  // for low-dimensional data, better to do more?
  int params_per = max_block_params;
  if (params->rows < max_block_params)
    params_per = next_pow2(params->rows, max_block_params);

  int data_per = max_threads / params_per;
  // at least 16 data points per block
  while (data_per < 16 & params_per > 1) {
    params_per /= 2;
    data_per *= 2;
  }

  while (1) {
    while (compute_shmem(data, params, params_per, data_per) > max_smem) {
      if (data_per <= 1)
        break;
      if (params_per > 1)
        params_per /= 2;
      else
        data_per /= 2;
    }
    // can't fit max_block_params sets of parameters into the shared memory
    if (data_per == 0) {
      data_per = 1;
      params_per /= 2;
      // start over the tuning
      continue;
    }
    else break;
  }

  // possible to squeeze more data?
  while (compute_shmem(data, params, params_per, 2 * data_per) <= max_smem)
    if (2 * data_per * params_per <= max_threads)
      data_per *= 2;
    else
      break;

  info->data_per_block = data_per;
  info->params_per_block = params_per;
}

__device__ int d_next_multiple(int k, int mult) {
  if (k % mult)
    return k + (mult - k % mult);
  else
    return k;
}

int next_multiple(int k, int mult) {
  if (k % mult)
    return k + (mult - k % mult);
  else
    return k;
}

__device__ float compute_pdf(float* data, float* params, int dim) {
  unsigned int LOGDET_OFFSET = dim * (dim + 3) / 2;
  float* mean = params;
  float* sigma = params + dim;
  float mult = params[LOGDET_OFFSET];
  float logdet = params[LOGDET_OFFSET + 1];

  float discrim = 0;
  float sum;
  unsigned int i, j;
  for (i = 0; i < dim; ++i)
  {
    sum = 0;
    for(j = 0; j <= i; ++j) {
      sum += *sigma++ * (data[j] - mean[j]);
    }
    discrim += sum * sum;
  }
  return log(mult) - 0.5 * (discrim + logdet + LOG_2_PI * dim);
}

__device__ void copy_data(const PMatrix* data, float* sh_data,
                          unsigned int thidx, unsigned int thidy, unsigned int obs_num)
{
  // if (obs_num >= data->rows)
  //   return;

  float val;
  for (unsigned int chunk = 0; chunk < data->cols; chunk += blockDim.y)
  {
    val = data->buf[data->stride * obs_num + chunk + thidy];
    if (chunk + thidy < data->cols) {
      sh_data[thidx * data->cols + chunk + thidy] = val;
    }
  }
  __syncthreads();
}

__device__ void copy_params(const PMatrix* params, float* sh_params,
                            unsigned int thidx, unsigned int thidy, unsigned int param_index)
{
  if (param_index >= params->rows)
    return;

  for (unsigned int chunk = 0; chunk < params->stride; chunk += blockDim.x)
  {
    if (chunk + thidx < params->stride)
      sh_params[thidy * params->stride + chunk + thidx] = \
        params->buf[params->stride * param_index + chunk + thidx];
  }
  __syncthreads();
}

__device__ void copy_chunks(float* in_buf, float* out_buf,
                            unsigned int tid, unsigned int total) {
  for (unsigned int chunk = 0; chunk + tid < total; chunk += blockDim.x) {
    out_buf[chunk + tid] = in_buf[chunk + tid];
  }
}

__global__ void mvnpdf_k(const PMatrix data, const PMatrix params,
                         const BlockDesign design, float* output) {

  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

  unsigned int rel_param = tid / design.data_per_block;
  unsigned int rel_data = tid - rel_param * design.data_per_block;

  unsigned int obs_num = design.data_per_block * blockIdx.x + rel_data;
  unsigned int param_num = design.params_per_block * blockIdx.y + rel_param;

  // set up shared data
  extern __shared__ float shared_data[];

  float* sh_params = shared_data; // store parameters
  float* sh_data = sh_params + design.params_per_block * params.stride; // store data
  float* sh_result = sh_data + design.data_per_block * data.stride; // store pdfs

  // copy_data(&data, sh_data, thidx, thidy, obs_num);
  // copy_params(&params, sh_params, thidx, thidy, param_index);

  copy_chunks(data.buf + design.data_per_block * blockIdx.x * data.stride,
              sh_data, tid,
              min(data.rows - design.data_per_block * blockIdx.x,
                  design.data_per_block) * data.stride);

  copy_chunks(params.buf + design.params_per_block * blockIdx.y * params.stride,
              sh_params, tid,
              min(design.params_per_block,
                  params.rows - design.params_per_block * blockIdx.y) * params.stride);

  __syncthreads();

  // allocated enough shared memory so that this will not walk out of bounds
  // no matter what, though some of the results will be garbage
  sh_result[tid] = compute_pdf(sh_data + rel_data * data.stride,
                               sh_params + rel_param * params.stride,
                               data.cols);
  __syncthreads();

  unsigned int result_idx = data.rows * param_num + obs_num;

  // output is column-major, so this will then coalesce
  if (obs_num < data.rows & param_num < params.rows) {
    output[result_idx] = sh_result[tid];
  }
}


/*
__device__ void _write_results(PMatrix* data, PMatrix* params,
                               float* output, float* sh_result,
                               int thidx, int thidy,
                               int tid)
{
  // write out in other order to coalesce
  // transpose! to get it to coalesce
  const int result_idx = param_index * data.rows + obs_num;

  // thread number in column-major order
  tid = thidx * blockDim.y + thidy;
  obs_num = blockDim.x * blockIdx.x + tid / blockDim.y;
  param_index = blockIdx.y * blockDim.y + tid % blockDim.y;
  const int result_idx = params.rows * obs_num + tid % blockDim.y;

  if (obs_num < data.rows & param_index < params.rows) {
     float d = compute_pdf(sh_data + thidx * data.cols,
                           sh_params + thidy * params.stride,
                           data.cols);
     sh_result[thidx * blockDim.x + thidy] = d;
  }
  __syncthreads();

  // int result_idx = params.rows * obs_num + param_index;
  int result_idx = (blockIdx.x * blockDim.x * params.rows
                     + blockIdx.y * blockDim.y + thidy * params.rows
                     + thidx);
  if (obs_num < data.rows & param_index < params.rows) {
     output[result_idx] = sh_result[thidx + thidy * blockDim.y];
  }
}
*/

// XXX: fix this
int MAX_BLOCK_PARAMS = 64;

cudaError_t invoke_mvnpdf(PMatrix data, PMatrix params, float* d_pdf) {
  // Need to automatically tune block / grid layout to maximize shared memory
  // usage and coalescence, reduce wasted threads!
  BlockDesign design;
  get_tuned_layout(&design, &data, &params, MAX_BLOCK_PARAMS);

  int nthreads = design.data_per_block * design.params_per_block;

  // Now set up grid layout / block size
  int grid_x = get_boxes(data.rows, design.data_per_block);
  int grid_y = get_boxes(params.rows, design.params_per_block);
  dim3 gridPDF(grid_x, grid_y);
  dim3 blockPDF(nthreads, 1);

  int sharedMemSize = compute_shmem(&data, &params,
                                    design.params_per_block,
                                    design.data_per_block);

#ifdef DEBUG
  printf("number params: %d, number data points: %d\n",
         design.params_per_block, design.data_per_block);
  printf("sharedMemSize: %d\n", sharedMemSize);
  printf("block: %d x %d, grid: %d x %d\n", blockPDF.x, blockPDF.y,
         gridPDF.x, gridPDF.y);
  printf("design: %d x %d\n", design.data_per_block, design.params_per_block);

  printf("nparams: %d\n", params.rows);
#endif

  mvnpdf_k<<<gridPDF,blockPDF,sharedMemSize>>>(data, params, design, d_pdf);
  return cudaSuccess;
}

void mvnpdf(float* h_data, /** Data-vector; padded */
             float* h_params, /** Density info; already padded */
             float* h_pdf, /** Resultant PDF */
             int data_dim,
             int total_obs,
             int nparams, // multiple sets of parameters
             int param_stride, // with padding
             int data_stride // with padding
  ) {

  float* d_data;
  float* d_params;
  float* d_pdf;
  cudaError_t error;

  PMatrix pdata, pparams;
  CATCH_ERR(cudaMalloc((void**) &d_pdf, total_obs * nparams * sizeof(float)));
  CATCH_ERR(cudaMalloc((void**) &d_data,
                       data_stride * total_obs * sizeof(float)));
  CATCH_ERR(cudaMalloc((void**) &d_params,
                       param_stride * nparams * sizeof(float)));

  h_to_d(h_data, d_data, total_obs * data_stride);
  h_to_d(h_params, d_params, nparams * param_stride);

  PMatrix_init(&pdata, d_data, total_obs, data_dim, data_stride);
  PMatrix_init(&pparams, d_params, nparams,
               data_dim * (data_dim + 3) / 2 + 2, param_stride);

  // printf("data dim: %d\n", pdata.cols);
  // printf("data padded dim: %d\n", pdata.stride);

  invoke_mvnpdf(pdata, pparams, d_pdf);
  d_to_h(d_pdf, h_pdf, total_obs * nparams);

  cudaFree(d_data);
  cudaFree(d_params);
  cudaFree(d_pdf);
}

void cpu_mvnormpdf(float* x, float* density, float * output, int D, int N, int T) {
    int LOGDET_OFFSET = D * (D + 3) / 2;
    int MEAN_CHD_DIM = D * (D + 3) / 2  + 2;

    int PACK_DIM = next_multiple(MEAN_CHD_DIM, 16);
    int DATA_PADDED_DIM = D; // next_multiple(D, 8);

    float* xx = (float*) malloc(D * sizeof(float));
    int obs, component;

    for (obs = 0; obs < N; obs++) {
        for (component = 0; component < T; component++) {
            float discrim;
            float* tData = x + obs * DATA_PADDED_DIM;
            float* tDensityInfo = density + component * PACK_DIM;
            float* tMean = tDensityInfo;
            float* tSigma = tDensityInfo + D;
            float  tP = tDensityInfo[LOGDET_OFFSET];
            float  tLogDet = tDensityInfo[LOGDET_OFFSET+1];

            // Do density calculation
            discrim = 0;
            for(int i=0; i<D; i++) {
                float sum = 0;
                for(int j=0; j<=i; j++) {
                  sum += *tSigma * (tData[j] - tMean[j]); // xx[j] is always calculated since j <= i
                  tSigma++;
                }

                discrim += sum * sum;
            }

            float d = log(tP) - 0.5 * (discrim + tLogDet + (LOG_2_PI*(float) D));
            output[obs * T + component] = d;
        }
    }
    free(xx);
}


#ifdef __cplusplus
}
#endif

#endif // _INCLUDED_MVNPDF
