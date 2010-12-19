#ifndef __MVNPDF_H__
#define __MVNPDF_H__

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "gpustats_common.h"

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

#endif
