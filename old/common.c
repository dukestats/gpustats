#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "common.h"

#if _WIN32
	#define isnan(x) ((x) != (x))
#endif

void set_device(int device) {
  cudaError_t error;
  CATCH_ERR(cudaSetDevice(device));
}

REAL *allocateGPURealMemory(int length) {
#ifdef DEBUG
	fprintf(stderr,"Entering ANMA-Real\n");
#endif

	REAL *data;
	cudaError_t error;
	SAFE_CUDA(cudaMalloc((void**) &data, SIZE_REAL * length),data);
	if (data == NULL) {
		fprintf(stderr,"Failed to allocate REAL (%d) memory on device!\n",
				length);
		// TODO clean up and gracefully die
		exit(-1);
	}

#ifdef DEBUG
	fprintf(stderr,"Allocated %d to %d.\n",data,(data +length));
	fprintf(stderr,"Leaving ANMA\n");
#endif

	return data;
}

INT *allocateGPUIntMemory(int length) {

#ifdef DEBUG
	fprintf(stderr,"Entering ANMA-Int\n");
#endif

	INT *data;
	cudaError_t error;
	SAFE_CUDA(cudaMalloc((void**) &data, SIZE_INT * length),data);
	if (data == NULL) {
		fprintf(stderr,"Failed to allocate INT memory on device!\n");
		exit(-1);
	}

#ifdef DEBUG
	fprintf(stderr,"Allocated %d to %d.\n",data,(data+length));
	fprintf(stderr,"Leaving ANMA\n");
#endif

	return data;
}

void freeGPUMemory(void *ptr) {

#ifdef DEBUG
	fprintf(stderr,"Entering FNMA\n");
#endif

	if (ptr != 0) {
		cudaFree(ptr);
	}

#ifdef DEBUG
	fprintf(stderr,"Leaving FNMA\n");
#endif
}

void storeGPURealMemoryArray(REAL *toGPUPtr, REAL *fromGPUPtr, int length) {
	cudaError_t error;
	SAFE_CUDA(cudaMemcpy(toGPUPtr, fromGPUPtr, SIZE_REAL*length, cudaMemcpyDeviceToDevice),toGPUPtr);
}

void storeGPUIntMemoryArray(INT *toGPUPtr, INT *fromGPUPtr, int length) {
	cudaError_t error;
	SAFE_CUDA(cudaMemcpy(toGPUPtr, fromGPUPtr, SIZE_INT*length, cudaMemcpyDeviceToDevice),toGPUPtr);
}
