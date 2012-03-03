/*
  Automatically-generated kernel for %(name)s

  For multivariate distributions, coordinates to utilize shared memory

  TODO: How to avoid bank conflicts
  TODO: How to ensure coalescence
 */

__global__ void k_%(name)s(float* g_output,
						   float* g_data,
						   float* g_params,
						   int data_per_block,
						   int params_per_block,
						   int data_rows,
						   int data_stride,
						   int data_cols,
						   int params_rows,
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
  float* sh_result = sh_data + data_per_block * data_stride;

  copy_chunks(g_data + data_per_block * blockIdx.x * data_stride,
              sh_data, tid,
              min(data_rows - data_per_block * blockIdx.x,
                  data_per_block) * data_stride);

  copy_chunks(g_params + params_per_block * blockIdx.y * params_stride,
              sh_params, tid,
              min(params_per_block,
                  params_rows - params_per_block * blockIdx.y) * params_stride);

  __syncthreads();

  // allocated enough shared memory so that this will not walk out of bounds
  // no matter what, though some of the results will be garbage
  sh_result[tid] = %(name)s(sh_data + rel_data * data_stride,
                            sh_params + rel_param * params_stride,
                            data_cols);
  __syncthreads();

  unsigned int result_idx = data_rows * param_num + obs_num;
  // unsigned int result_idx = obs_num * data_cols + param_num

  // g_output is column-major, so this will then coalesce
  if (obs_num < data_rows & param_num < params_rows) {
    g_output[result_idx] = sh_result[tid];
  }
}

// foo
