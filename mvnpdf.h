#ifndef __MVNPDF_H__
#define __MVNPDF_H__

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "common.h"

cudaError_t gpuMvNormalPDF(
  REAL* inData, /** Data-vector; padded */
  REAL* inDensityInfo, /** Density info; already padded */
  REAL* outPDF, /** Resultant PDF */
  int iD,
  int iN,
  int iTJ,
  int PACK_DIM,
  int DIM
  );

void cpu_mvnormpdf(float* x, float* density, float * output, int D,
				   int N, int T);

void testf(float* ptr, int n);

#endif
