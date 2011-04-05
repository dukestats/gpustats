// If compute cabability is 1.1 or 1.0 ... this really should be padded in T to a multiple of 16.
// If    "        "        is higher, this should be pretty good because of caching and the
// ability to read mis-alighned sequences of memory in 2 or less reads..



__global__ void k_%(name)s(float* in_measure, /** Precomputed measure */
			   float* in_random, /** Precomputed random number */
			   int* out_component, /** Resultant choice */
			   int iN, int iT) {
  //const int iN = dims[0];
  //const int iT = dims[1];

  const int sample_density_block = blockDim.x;
  const int sample_block = blockDim.y;
  const int thidx = threadIdx.x;
  const int thidy = threadIdx.y;
  const int datumIndex = blockIdx.x * sample_block  + thidy;
  const int pdfIndex = datumIndex * iT;
  const int tid = thidy*sample_density_block + thidx;
 
  
  //__shared__ REAL measure[sample_block][sample_density_block]; 
  //__shared__ REAL sum[sample_block];

  // Make block size flexible ... 
  extern __shared__ float shared_data[];
  float* measure = shared_data; // sample_block by sample_density_block
  float* sum = measure + sample_block*sample_density_block;
  float* work = sum + sample_block;

  // use 'work' in multiple places to save on memory
  if (thidx == 0) {
    sum[thidy] = 0;
#if defined(LOGPDF)
    work[thidy] = -10000;
#else 
    work[thidy] = 0;
#endif
  }

#if defined(LOGPDF)
  //get the max values
  for(int chunk = 0; chunk < iT; chunk += sample_density_block) {
    measure[thidy*sample_block + thidx] = in_measure[pdfIndex + chunk + thidx];
    __syncthreads();
    
    if (thidx == 0) {
      for(int i=0; i<sample_density_block; i++) {
	if(chunk + i < iT){
	  float dcurrent = measure[thidy*sample_block + i];
	  if (dcurrent > work[thidy]) {
	    work[thidy] = dcurrent;
	  }
	}
      }
    }
    __syncthreads();
  }
#endif


  //get scaled cummulative pdfs
  for(int chunk = 0; chunk < iT; chunk += sample_density_block) {

    measure[thidy*sample_block + thidx] = in_measure[pdfIndex + chunk + thidx];
    
    __syncthreads();

    if (thidx == 0) {
      for(int i=0; i<sample_density_block; i++) {
	if (chunk + i < iT){
#if defined(LOGPDF)
	  //rescale and exp()
	  sum[thidy] += exp(measure[thidy*sample_block + i] - work[thidy]); 
#else
	  sum[thidy] += measure[thidy*sample_block + i];
#endif
	  measure[thidy*sample_block + i] = sum[thidy];
	}
      }
    }

    if (chunk + thidx < iT)
      in_measure[pdfIndex + chunk + thidx] = measure[thidy*sample_block + thidx];
    
    __syncthreads();
  }

#if defined(LOGPDF)
  if (thidx == 0){
    work[thidy] = 0;
  }
#endif

  float* randomNumber = sum;
  const int result_id = blockIdx.x * sample_block + tid;
  if ( tid < sample_block )
    randomNumber[tid] = in_random[result_id] * sum[tid];

  // randomNumber = in_random[datumIndex] * sum[thidy];

  // Find the right bin for the random number ...
  for(int chunk = 0; chunk < iT; chunk += sample_density_block) {
    
    measure[thidy*sample_block + thidx] = in_measure[pdfIndex + chunk + thidx];
    __syncthreads();

    if (thidx == 0) {

      // storing the index in a float is better because it avoids
      // bank conflicts ... 
      for(int i=0; i<sample_density_block; i++) {
	if (chunk + i < iT){
	  if (randomNumber[thidy] > measure[thidy*sample_block + i]){
	    work[thidy] = i + chunk + 1;
	  }
	}
      }
      if ((int) work[thidy] >= iT) {work[thidy] = iT-1;}
    }
  }
  __syncthreads();

  // this is now coalesced
  if (result_id < iN && tid < sample_block)
    out_component[result_id] = (int) work[tid];

  //if (thidx == 0 && datumIndex < iN) 
  //  out_component[datumIndex] = work;


}



