/*
  Block layout : npmfs x nhelpers
  Grid layout : K x 1
  K is the smallest number such that K * npmfs >= pmf_rows
 */

__global__ void
k_%(name)s(float* g_pmf, /** Precomputed pmf */
		   float* g_urand, /** Precomputed random number */
		   float* g_output, /** Resultant choice */
		   int pmf_rows,
		   int pmf_cols,
		   int pmf_stride
  ) {
  // blockDim.x = number of pmfs sampled from in this block
  // blockDim.y = number of helper threads per pmf
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int thidx = threadIdx.x;
  unsigned int npmfs = blockDim.x;

  // Make block size flexible ...
  extern __shared__ float shared_data[];

  float* sh_pmf = shared_data; // npmfs * pmf_stride floats
  float* sh_work = sh_pmf + npmfs * pmf_stride; // nmpfs floats

  // Move pmf data into shared memory
  copy_chunks(g_pmf + npmfs * pmf_stride * blockIdx.x,
			  sh_pmf, tid,
			  min(npmfs,
				  pmf_rows - npmfs * blockIdx.x) * pmf_stride);
  __syncthreads();

  // move uniform random draws into shared memory
  copy_chunks(g_urand + npmfs * blockIdx.x,
			  sh_work, tid,
			  min(npmfs, pmf_rows - npmfs * blockIdx.x));
  __syncthreads();

  // done copying, now move pointer to start of pmf for this row of threads
  sh_pmf = sh_pmf + thidx * pmf_stride;

  // compute normalizing constant using atomic operators?

  // for(int chunk = 0; chunk + thidx < pmf_cols; chunk += blockDim.x) {
  // 	atomic_add(sh_work + thidy, sh_pmf[chunk + thidx]);
  // }

  if (threadIdx.y == 0 && thidx < pmf_rows - npmfs * blockIdx.x) {
	float norm_const = 0;
  	for (int i = 0; i < pmf_cols; ++i) {
  	  norm_const += sh_pmf[i];
  	}

	float draw = sh_work[thidx];

	// replace with scaled cumulative pdf
	sh_pmf[0] /= norm_const;
	sh_work[thidx] = 0;
	if (sh_pmf[0] < draw) {
	  for(int i = 1; i < pmf_cols; i++) {
		sh_pmf[i] = sh_pmf[i-1] + sh_pmf[i] / norm_const;
		if (sh_pmf[i] >= draw) {
		  sh_work[thidx] = i;
		  break;
		}
	  }
	}
  }
  __syncthreads();

  // this is now coalesced
  unsigned int result_id = blockIdx.x * npmfs + tid;
  if (result_id < pmf_rows && tid < npmfs)
    g_output[result_id] = sh_work[tid];

  return;
}
