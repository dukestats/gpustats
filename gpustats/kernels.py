
log_pdf_mvnormal = """
__device__ float pdf_mvnormal(float* data, float* params, int dim) {
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

log_pdf_normal = """
__device__ float log_pdf_normal(float x, float* params) {
  float mean = params[0];
  float std = params[1]

  // standardize
  x = (x - mean) / std;

  return - x * x / 2 - 0.5 * LOG_2_PI - log(std);
}
"""

pdf_normal = """
__device__ float pdf_normal(float x, float* params) {
  return exp(log_pdf_normal(x, params));
}
"""
