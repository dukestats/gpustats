#ifndef _INCLUDED_MVNPDF
#define _INCLUDED_MVNPDF

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "gpustats_common.h"

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
    if (isLogScaled>0) {
        d = log(tP)-0.5 * (discrim + tLogDet);
    } else {
        REAL mydim = (REAL)iD;
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

cudaError_t gpuMvNormalPDF(
                    REAL* inData, /** Data-vector; padded */
                    REAL* inDensityInfo, /** Density info; already padded */
                    REAL* outPDF, /** Resultant PDF */
                    int iD,
                    int iN,
                    int iTJ,
					int PACK_DIM,
					int DIM
                    ) {

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
        mvNormalPDF<<<gridPDF,blockPDF,sharedMemSize>>>(inData,inDensityInfo,outPDF,iD, iN, iTJ,1);
    #else
        mvNormalPDF<<<gridPDF,blockPDF,sharedMemSize>>>(inData,inDensityInfo,outPDF,iD, iN, iTJ,0);
    #endif
    return cudaSuccess;
}


#endif // _INCLUDED_MVNPDF
