#ifndef __MVNPDF_H__
#define __MVNPDF_H__

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "common.h"

// Simple strided matrix data structure, far as I can tell there's little or no
// overhead in the compiled version.
typedef struct PMatrix {
  float* buf; // C-style row-major data
  int rows; // actual number of rows
  int cols; // actual number of columns
  int stride; // data length of row
} PMatrix;

void PMatrix_init(PMatrix* mat, float* data, int rows, int cols, int stride){
  mat->buf = data;
  mat->rows = rows;
  mat->cols = cols;
  mat->stride = stride;
}

void mvnpdf2(float* h_data, /** Data-vector; padded */
			 float* h_params, /** Density info; already padded */
			 float* h_pdf, /** Resultant PDF */
			 int data_dim,
			 int total_obs,
			 int nparams,
			 int param_stride, // with padding
			 int data_stride // with padding
  );

void cpu_mvnormpdf(float* x, float* density, float * output, int D,
				   int N, int T);

void testf(float* ptr, int n);

#endif
