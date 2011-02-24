from gpustats.codegen import MVDensityKernel, DensityKernel, Exp
import gpustats.codegen as cg

# TODO: check for name conflicts!

_log_pdf_mvnormal = """
__device__ float %(name)s(float* data, float* params, int dim) {
  unsigned int LOGDET_OFFSET = dim * (dim + 3) / 2;
  float* mean = params;
  float* sigma = params + dim;
  float mult = params[LOGDET_OFFSET];
  float logdet = params[LOGDET_OFFSET + 1];

  float discrim = 0;
  float sum;
  unsigned int i, j;
  for (i = 0; i < dim; ++i)
  {
    sum = 0;
    for(j = 0; j <= i; ++j) {
      sum += *sigma++ * (data[j] - mean[j]);
    }
    discrim += sum * sum;
  }
  return log(mult) - 0.5 * (discrim + logdet + LOG_2_PI * dim);
}
"""
log_pdf_mvnormal = MVDensityKernel('log_pdf_mvnormal', _log_pdf_mvnormal)
pdf_mvnormal = Exp('pdf_mvnormal', log_pdf_mvnormal)


_log_pdf_normal = """
__device__ float %(name)s(float* x, float* params) {
  // mean stored in params[0]
  float std = params[1];

  // standardize
  float xstd = (*x - params[0]) / std;
  return - (xstd * xstd) / 2 - 0.5 * LOG_2_PI - log(std);
}
"""
log_pdf_normal = DensityKernel('log_pdf_normal', _log_pdf_normal)
pdf_normal = Exp('pdf_normal', log_pdf_normal)
