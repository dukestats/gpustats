#ifndef _INCLUDED_MVNPDF
#define _INCLUDED_MVNPDF

#ifdef __cplusplus
extern "C" {
#endif

#include "mvnpdf.h"

#define BLOCK_SIZE 16
#define BLOCK_TOTAL 256

void inline h_to_d(float* h_ptr, float* d_ptr, int n){
  cudaMemcpy(d_ptr, h_ptr, n * sizeof(float), cudaMemcpyHostToDevice);
}

void inline d_to_h(float* d_ptr, float* h_ptr, int n){
  cudaMemcpy(h_ptr, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
}

__device__ int next_multiple(int k, int mult) {
  if (k % mult)
	return k + (mult - k % mult);
  else
	return k;
}

typedef struct {
  REAL* data;
  int rows;
  int pad_columns;
  int columns;
  int stride;
} PMatrix;

/* Thread-Block design:
 * 1 thread per datum*density
 * Block grid(DATA_IN_BLOCK,DENSITIES_IN_BLOCK)
 * DATA_IN_BLOCK = # of datum per block
 * DENSITIES_IN_BLOCK = # of densities per block
 */
#define TWISTED_DENSITY
__global__ void mvNormalPDF(
                    REAL* inData, /** Data-vector; padded */
                    REAL* inDensityInfo, /** Density info; already padded */
                    REAL* outPDF, /** Resultant PDF */
                    int iD,
                    int iN,
                    int iTJ,
                    int isLogScaled
                ) {
    const int thidx = threadIdx.x;
    const int thidy = threadIdx.y;

    const int dataBlockIndex = blockIdx.x * DATA_IN_BLOCK;
    const int datumIndex = dataBlockIndex + thidx;

    const int densityBlockIndex = blockIdx.y * DENSITIES_IN_BLOCK;
    const int densityIndex = densityBlockIndex + thidy;

    #if defined(TWISTED_DENSITY)
        const int pdfIndex = blockIdx.x * DATA_IN_BLOCK * iTJ +
            blockIdx.y * DENSITIES_IN_BLOCK + thidy * iTJ + thidx;
    #else
        const int pdfIndex = datumIndex * iTJ + densityIndex;
    #endif

    extern __shared__ REAL sData[];
    REAL *densityInfo = sData;
    // do this for now, will be more efficient to pass them in as parameters?
    //-------------------------------------------------------
    int LOGDET_OFFSET = iD * (iD + 3) / 2;
    int MEAN_CHD_DIM = iD * (iD + 3) / 2    + 2;
    int PACK_DIM = 16;
    while (MEAN_CHD_DIM > PACK_DIM) {PACK_DIM += 16;}
    int DATA_PADDED_DIM = BASE_DATAPADED_DIM;
    while (iD > DATA_PADDED_DIM) {DATA_PADDED_DIM += BASE_DATAPADED_DIM;}
    //--------------------------------------------------

    const int data_offset = DENSITIES_IN_BLOCK * PACK_DIM;
    REAL *data = &sData[data_offset];

    #if defined(TWISTED_DENSITY)
        REAL *result_trans = &sData[data_offset+DATA_IN_BLOCK * iD];
    #endif

    //Read in data
    for(int chunk = 0; chunk < iD; chunk += DENSITIES_IN_BLOCK)
    if (chunk + thidy < iD ) {
        data[thidx * iD + chunk + thidy] = inData[DATA_PADDED_DIM*datumIndex + chunk + thidy];
    }


    // Read in density info by chunks
    for(int chunk = 0; chunk < PACK_DIM; chunk += DATA_IN_BLOCK) {
        if (chunk + thidx < PACK_DIM) {
            densityInfo[thidy * PACK_DIM + chunk + thidx] = inDensityInfo[PACK_DIM*densityIndex + chunk + thidx];
        }
    }
    __syncthreads();

    // Setup pointers
    REAL* tData = data+thidx*iD;
    REAL* tDensityInfo = densityInfo + thidy * PACK_DIM;


    REAL* tMean = tDensityInfo;         //do we need to unallocate shared/register variables?
    REAL* tSigma = tDensityInfo + iD;
    REAL  tP = tDensityInfo[LOGDET_OFFSET];
    REAL  tLogDet = tDensityInfo[LOGDET_OFFSET+1];

    // Do density calculation
    REAL discrim = 0;
    for(int i=0; i<iD; i++) {
        REAL sum = 0;
        for(int j=0; j<=i; j++) {
            sum += *tSigma++ * (tData[j] - tMean[j]); // xx[j] is always calculated since j <= i
        }
        discrim += sum * sum;
    }
    REAL d;
	REAL mydim = (REAL)iD;
    if (isLogScaled>0) {
	  d = log(tP)-0.5 * (discrim + tLogDet + (LOG_2_PI * mydim));
    } else {
	  d = tP * exp(-0.5 * (discrim + tLogDet + (LOG_2_PI*mydim)));
    }
    #if defined(TWISTED_DENSITY)
        result_trans[thidx * DATA_IN_BLOCK + thidy] = d;
        __syncthreads();
    #endif


    if (datumIndex < iN & densityIndex < iTJ) {
        #if defined(TWISTED_DENSITY)
            outPDF[pdfIndex] = result_trans[thidx + thidy * DENSITIES_IN_BLOCK];
        #else

            outPDF[pdfIndex] = d;
        #endif
    }
}

__device__ float compute_pdf(float* data, float* params, int iD) {
  const int LOGDET_OFFSET = iD * (iD + 3) / 2;
  float* mean = params;
  float* sigma = params + iD;
  float mult = params[LOGDET_OFFSET];
  float logdet = params[LOGDET_OFFSET + 1];

  float discrim = 0;
  float sum;

  for (int i = 0; i < iD; ++i)
  {
   	sum = 0;
   	for(int j=0; j <= i; j++) {
   	  sum += *sigma++ * (data[j] - mean[j]);
   	}
   	discrim += sum * sum;
  }

  return log(mult) - 0.5 * (discrim + logdet + LOG_2_PI * (float) iD);
}

__global__ void mvNormalPDF2(
                    float* glob_data, /** Data-vector; padded */
                    float* glob_params, /** Density info; already padded */
                    float* outPDF, /** Resultant PDF */
                    int iD,
                    int iN
                ) {
  int PACK_DIM = next_multiple(iD * (iD + 3) / 2 + 2, 16);
  int DATA_PADDED_DIM = next_multiple(iD, BASE_DATAPADED_DIM);

  const int block_start = blockIdx.x * blockDim.x * blockDim.y;
  const int data_offset = threadIdx.x * blockDim.x + threadIdx.y;
  const int obs_num = block_start + data_offset;
  const int data_start = obs_num * DATA_PADDED_DIM;
  const int rel_data_start = data_offset * DATA_PADDED_DIM;

  extern __shared__ float sData[];

  float* sh_params = sData;
  float* sh_data = sData + PACK_DIM;

  for (int i = data_start; i < data_start + iD; ++i) {
	sh_data[i - block_start * DATA_PADDED_DIM] = glob_data[data_start + i];
  }

  // read mean, cov, scalar, logdet into shared memory
  // TODO: make sure whole thing read into memory
  if (data_offset < PACK_DIM) {
	sh_params[data_offset] = glob_params[data_offset];
  }
  __syncthreads();

  float density = compute_pdf(sh_data + rel_data_start,
							  sh_params, iD);

  __syncthreads();

  if (obs_num < iN) {
	outPDF[obs_num] = density;
  }
}

cudaError_t gpuMvNormalPDF2(
                    float* hData, /** Data-vector; padded */
                    float* hParams, /** Density info; already padded */
                    float* hPDF, /** Resultant PDF */
                    int iD,
                    int iN,
					int PACK_DIM,
					int PADDED_DATA_DIM
                    ) {

  float* dData;
  float* dParams;
  float* dPDF;

  cudaMalloc(&dData, PADDED_DATA_DIM * iN * sizeof(float));
  cudaMalloc(&dParams, PACK_DIM * sizeof(float));
  cudaMalloc(&dPDF, iN * sizeof(float));

  h_to_d(hData, dData, iN * PADDED_DATA_DIM);
  h_to_d(hParams, dParams, PACK_DIM);

  dim3 gridPDF(iN/BLOCK_TOTAL, 1);
  if (iN % BLOCK_TOTAL != 0)
	gridPDF.x += 1;
  dim3 blockPDF(BLOCK_SIZE, BLOCK_SIZE);
  int sharedMemSize = PACK_DIM * SIZE_REAL + PADDED_DATA_DIM * BLOCK_SIZE * BLOCK_SIZE;
  mvNormalPDF2<<<gridPDF,blockPDF,sharedMemSize>>>(dData, dParams, dPDF, iD, iN);

  d_to_h(dPDF, hPDF, iN);

  cudaFree(dData);
  cudaFree(dParams);
  cudaFree(dPDF);

  return cudaSuccess;
}


cudaError_t gpuMvNormalPDF(
                    REAL* hData, /** Data-vector; padded */
                    REAL* hParams, /** Density info; already padded */
                    REAL* hPDF, /** Resultant PDF */
                    int iD,
                    int iN,
                    int iTJ,
					int PACK_DIM,
					int DIM
                    ) {

  float* dData;
  float* dParams;
  float* dPDF;

  cudaMalloc(&dData, DIM * iN * sizeof(float));
  cudaMalloc(&dParams, PACK_DIM * sizeof(float));
  cudaMalloc(&dPDF, iN * sizeof(float));

  h_to_d(hData, dData, iN);
  h_to_d(hParams, dParams, PACK_DIM);

  dim3 gridPDF(iN/DATA_IN_BLOCK, iTJ/DENSITIES_IN_BLOCK);
  if (iN % DATA_IN_BLOCK != 0)
	gridPDF.x += 1;
  if (iTJ % DENSITIES_IN_BLOCK != 0)
	gridPDF.y += 1;

  dim3 blockPDF(DATA_IN_BLOCK,DENSITIES_IN_BLOCK);
#if defined(TWISTED_DENSITY)
  int sharedMemSize = (DENSITIES_IN_BLOCK * PACK_DIM + DATA_IN_BLOCK * DIM \
					   + DENSITIES_IN_BLOCK*DATA_IN_BLOCK) * SIZE_REAL;
#else
  int sharedMemSize = (DENSITIES_IN_BLOCK * PACK_DIM + DATA_IN_BLOCK * DIM) * SIZE_REAL;
#endif
#if defined(LOGPDF)
  mvNormalPDF<<<gridPDF,blockPDF,sharedMemSize>>>(dData, dParams, dPDF,iD, iN, iTJ,1);
#else
  mvNormalPDF<<<gridPDF,blockPDF,sharedMemSize>>>(dData, dParams, dPDF, iD, iN, iTJ,0);
#endif

  d_to_h(dPDF, hPDF, iN);

  cudaFree(dData);
  cudaFree(dParams);
  cudaFree(dPDF);

    return cudaSuccess;
}

void cpu_mvnormpdf(float* x, float* density, float * output, int D, int N, int T) {
    int LOGDET_OFFSET = D * (D + 3) / 2;
	int MEAN_CHD_DIM = D * (D + 3) / 2	+ 2;
	int PACK_DIM = 16;

	while (MEAN_CHD_DIM > PACK_DIM) {PACK_DIM += 16;}
	int DATA_PADDED_DIM = 8;
	while (D > DATA_PADDED_DIM) {DATA_PADDED_DIM += 8;}

    float* xx = (float*) malloc(D * sizeof(float));
    float mydim = (float) D;
    int data,component;

    for (data = 0; data < N; data++) {
        for (component = 0; component < T; component++) {
            float discrim;
            float* tData = x + data * DATA_PADDED_DIM;
            float* tDensityInfo = density + component * PACK_DIM;

            float* tMean = tDensityInfo;			//do we need to unallocate shared/register variables?
            float* tSigma = tDensityInfo + D;
            float  tP = tDensityInfo[LOGDET_OFFSET];
            float  tLogDet = tDensityInfo[LOGDET_OFFSET+1];

            // Do density calculation
            discrim = 0;
            for(int i=0; i<D; i++) {
                float sum = 0;
                for(int j=0; j<=i; j++) {
                    sum += *tSigma++ * (tData[j] - tMean[j]); // xx[j] is always calculated since j <= i
                }
                discrim += sum * sum;
            }

            float d = log(tP) - 0.5 * (discrim + tLogDet + (LOG_2_PI*mydim));
			// printf("discrim: %f\n", discrim);
			// printf("tP: %f\n", tP);
			// printf("tLogDet: %f\n", tLogDet);
			// printf("d: %f\n", d);
			// printf("idx: %d\n", data * T + component);
            output[data * T + component] = d;
        }
    }
	free(xx);
}


#ifdef __cplusplus
}
#endif

#endif // _INCLUDED_MVNPDF
