/*
  Automatically-generated kernel for %(name)s

  For multivariate distributions, coordinates to utilize shared memory

  TODO: How to avoid bank conflicts
  TODO: How to ensure coalescence
 */

__global__ void k_%(name)s(float* output,
			   float* data,
			   float* params,
			   int data_per_block,
			   int params_per_block,
			   int data_rows,
			   int data_stride,
			   int data_cols,
			   int params_rows,
			   int params_stride) {

  // Think of a more elegant, efficient way of doing this
  // use shared memory?
  //unsigned int data_per_block, params_per_block;
  //unsigned int data_rows, data_stride, data_cols;
  //unsigned int params_rows, params_stride;

  // inelegant, perhaps...
  //data_per_block = design[0];
  //params_per_block = design[1];
  //data_rows = design[2];
  //data_stride = design[3];
  //data_cols = design[4];
  //params_rows = design[5];
  //params_stride = design[6];
  // unsigned int params_cols = design[7];

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

  copy_chunks(data + data_per_block * blockIdx.x * data_stride,
              sh_data, tid,
              min(data_rows - data_per_block * blockIdx.x,
                  data_per_block) * data_stride);

  copy_chunks(params + params_per_block * blockIdx.y * params_stride,
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

  // output is column-major, so this will then coalesce
  if (obs_num < data_rows & param_num < params_rows) {
    output[result_idx] = sh_result[tid];
  }
}

// foo
