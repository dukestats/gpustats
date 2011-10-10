/*
  Automatically-generated kernel for %(name)s

  For univariate distributions
 */

__global__ void k_%(name)s(float* output,
                           float* data,
                           float* params,
                           int data_per_block,
                           int params_per_block,
                           int nobs,
                           int nparams,
                           int params_stride) {

  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

  unsigned int rel_param = tid / data_per_block;
  unsigned int rel_data = tid - rel_param * data_per_block;

  unsigned int obs_num = data_per_block * blockIdx.x + rel_data;
  unsigned int param_num = params_per_block * blockIdx.y + rel_param;

  // set up shared data
  extern __shared__ float shared_data[];
  float* sh_params = shared_data;
  float* sh_data = sh_params + params_per_block * params_stride;
  float* sh_result = sh_data + data_per_block;

  copy_chunks(data + data_per_block * blockIdx.x,
              sh_data, tid,
              min(nobs - data_per_block * blockIdx.x,
                  data_per_block));

  copy_chunks(params + params_per_block * blockIdx.y * params_stride,
              sh_params, tid,
              min(params_per_block,
                  nparams - params_per_block * blockIdx.y) * params_stride);

  __syncthreads();

  // allocated enough shared memory so that this will not walk out of bounds
  // no matter what, though some of the results will be garbage
  sh_result[tid] = %(name)s(sh_data + rel_data,
                            sh_params + rel_param * params_stride);
  __syncthreads();

  unsigned int result_idx = nobs * param_num + obs_num;

  // output is column-major, so this will then coalesce
  if (obs_num < nobs & param_num < nparams) {
    // output[result_idx] = obs_num;
    output[result_idx] = sh_result[tid];
  }
}
