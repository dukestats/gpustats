#ifndef __GPUSTATS_COMMON__
#define __GPUSTATS_COMMON__

#include <math.h>
#include <cuda.h>
/* Dimension specific definitions to ensure coalesced memory transactions */

extern int DIM,MEAN_CHD_DIM,PACK_DIM,CHD_DIM,LOGDET_OFFSET,DATA_PADDED_DIM,NCHUNKSIZE;

#define DENSITIES_IN_BLOCK		16 //4 //4 for 27d data, 16 for other data
#define	DATA_IN_BLOCK			16	//need >= 16 to be efficient
#define SAMPLE_BLOCK			32
#define SAMPLE_DENSITY_BLOCK	16

#define BASE_DATAPADED_DIM		8

#define SIGMA_BLOCK_SIZE		128
#define SIGMA_THREAD_SUM_SIZE		25

//#define LOGPDF

//#define CHECK_GPU

// For algorithm 2
#define	PAD_CSR				0		// Little (no?) performance gain on 9400M and complicates algorithm
#define PAD					1		// Removes some bank conflicts (?)
#define BLOCK_SIZE_COL		16		// # of data columns to process per block
#define BLOCK_SIZE_ROW 		32		// BLOCK_SIZE_ROW / HALFWARP = # of rows (components) to process per block
#define HALFWARP_LOG2		4
#define HALFWARP 			(1<<HALFWARP_LOG2)
#define GROW_INDICES		16

#define COMPACT_BLOCK	256

/* Definition of REAL can be switched between 'double' and 'float' */
#ifdef DOUBLE_PRECISION
	#define REAL		double
#else
	#define REAL		float
#endif

#define SIZE_REAL	sizeof(REAL)

#define INT			int

#define SIZE_INT	sizeof(INT)


/* Error codes */
#define CUDA_ERROR	1
#define CUDA_SUCCESS	0

#define SAFE_CUDA(call,ptr)		cudaError_t error = call; \
								if( error != 0 ) { \
									fprintf(stderr,"Error %s\n", cudaGetErrorString(error)); \
									fprintf(stderr,"Ptr = %d\n",ptr); \
									exit(-1); \
								}


#define MEMCPY(to,from,length,toType) { int m; \
										for(m=0; m<length; m++) { \
											to[m] = (toType) from[m]; \
										} }



#define LOG_2_PI 1.83787706640935
#define LOG_PI 1.144729885849400

int initCUDAContext();

int migrateContext(CUcontext context);

REAL *allocateGPURealMemory(int length);

INT  *allocateGPUIntMemory(int length);

void checkCUDAError(const char *msg);

void freeGPUMemory(void *ptr);

void storeGPURealMemoryArray(REAL *toGPUPtr, REAL *fromGPUPtr, int length);

void storeGPUIntMemoryArray(INT *toGPUPtr, INT *fromGPUPtr, int length);

void printfCudaVector(REAL *dPtr, int length);

void printfCudaInt(int *dPtr, int length);

void printfVectorD(double *ptr, int length);

void printfVectorF(float *ptr, int length);

void printfVector(REAL *ptr, int length);

void printfInt(int *ptr,int length);

REAL sumCudaVector(REAL *dPtr, int length);

int checkZeros(REAL* dPtr, int length);

void loadTipPartials(int instance);

void doStore(int instance);

void doRestore(int instance);

void handleStoreRestoreQueue(int instance);

const int MAX_GPU_COUNT = 8;


#endif // __GPUSTATS_COMMON__
