#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <cuda_runtime_api.h>

#include "common.h"

__global__ void mvNormalPDF(
					REAL* iData, /** Data-vector; padded */
					REAL* iDensityInfo, /** Density info; already padded */
					REAL* oMeasure, /** Resultant measure */
					int iD, /** Not currently necessary, as DIM is hardcoded */
					int iN,
					int iTJ,
					int isLogScaled
				);

cudaError_t gpuMvNormalPDF(
					REAL* iData, /** Data-vector; padded */
					REAL* iDensityInfo, /** Density info; already padded */
					REAL* oMeasure, /** Resultant measure */
					int iD, /** Not currently necessary, as DIM is hardcoded */
					int iN,
					int iTJ
				);

#endif // __KERNELS_H__
