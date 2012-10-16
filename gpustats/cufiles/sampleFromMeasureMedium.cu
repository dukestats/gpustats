// Original written by Marc Suchard 
// Modified by Andrew Cron 

__global__ void k_%(name)s(float* in_measure, /** Precomputed measure */
	   		   float* in_random, /** Precomputed random number */
			   int* out_component, /** Resultant choice */
			   int iN, int iT, int logged) {

  const int sample_density_block = blockDim.x;
  const int sample_block = blockDim.y;
  const int thidx = threadIdx.x;
  const int thidy = threadIdx.y;
  const int datumIndex = blockIdx.x * sample_block + thidy;
  const int pdfIndex = datumIndex * iT;
  const int tid = thidy*sample_density_block + thidx;
  const int stride = sample_density_block+1;

  // Make block size flexible ...
  extern __shared__ float shared_data[];
  float* measure = shared_data; // sample_block by stride
  float* sum = measure + sample_block*stride;
  float* work = sum + sample_block;

  // use 'work' in multiple places to save on memory
  if (tid < sample_block) {
    sum[tid] = 0;
    if(logged==1){
        work[tid] = -10000;
    } else {
        work[tid] = 0;
    }
  }


  if(logged==1){
  //get the max values
  for(int chunk = 0; chunk < iT; chunk += sample_density_block) {
    if(pdfIndex + chunk + thidx < iN*iT)
       measure[thidy*stride + thidx] = in_measure[pdfIndex + chunk + thidx];
    __syncthreads();

    if (tid < sample_block) {
      for(int i=0; i<sample_density_block; i++) {
    if(chunk + i < iT){
      float dcurrent = measure[tid*stride + i];
      if (dcurrent > work[tid]) {
        work[tid] = dcurrent;
      }
    }
      }
    }
    __syncthreads();
  }
  }


  //get scaled cummulative pdfs
  for(int chunk = 0; chunk < iT; chunk += sample_density_block) {
    if(pdfIndex + chunk + thidx < iN*iT)
       measure[thidy*stride + thidx] = in_measure[pdfIndex + chunk + thidx];

    __syncthreads();

    if (tid < sample_block) {
      for(int i=0; i<sample_density_block; i++) {
    if (chunk + i < iT){
      if(logged==1){
      //rescale and exp()
      sum[tid] += expf(measure[tid*stride + i] - work[tid]);
      } else {
      sum[tid] += measure[tid*stride + i];
      }
      measure[tid*stride + i] = sum[tid];
    }
      }
    }

    __syncthreads();

    if(datumIndex < iN && chunk + thidx < iT)
      in_measure[pdfIndex + chunk + thidx] = measure[thidy*stride + thidx];
 
  }

  __syncthreads();  

  if (tid < sample_block && logged==1){
    work[tid] = 0;
  }


  float* randomNumber = sum;
  const int result_id = blockIdx.x * sample_block + tid;
  if ( result_id < iN && tid < sample_block)
    randomNumber[tid] = in_random[result_id] * sum[tid];

  // Find the right bin for the random number ...
  for(int chunk = 0; chunk < iT; chunk += sample_density_block) {
    if(pdfIndex + chunk + thidx < iN*iT)
       measure[thidy*stride + thidx] = in_measure[pdfIndex + chunk + thidx];
    __syncthreads();

    if (tid < sample_block) {

      // storing the index in a float is better because it avoids
      // bank conflicts ...
      for(int i=0; i<sample_density_block; i++) {
    if (chunk + i < iT){
      if (randomNumber[tid] > measure[tid*stride + i]){
        work[tid] = i + chunk + 1;
      }
    }
      }
      if ( work[tid] >= iT) {work[tid] = iT-1;}
    }
    __syncthreads();
  }

  // this is now coalesced
  if (result_id < iN && tid < sample_block)
    out_component[result_id] = (int) work[tid];

}



