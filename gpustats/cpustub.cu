int MAX_BLOCK_PARAMS = 64;

cudaError_t invoke_mvnpdf(PMatrix data, PMatrix params, float* d_pdf) {
  // Need to automatically tune block / grid layout to maximize shared memory
  // usage and coalescence, reduce wasted threads!
  BlockDesign design;
  get_tuned_layout(&design, &data, &params, MAX_BLOCK_PARAMS);

  int nthreads = design.data_per_block * design.params_per_block;

  // Now set up grid layout / block size
  int grid_x = get_boxes(data.rows, design.data_per_block);
  int grid_y = get_boxes(params.rows, design.params_per_block);
  dim3 gridPDF(grid_x, grid_y);
  dim3 blockPDF(nthreads, 1);

  int sharedMemSize = compute_shmem(&data, &params,
                                    design.params_per_block,
                                    design.data_per_block);

#ifdef DEBUG
  printf("number params: %d, number data points: %d\n",
         design.params_per_block, design.data_per_block);
  printf("sharedMemSize: %d\n", sharedMemSize);
  printf("block: %d x %d, grid: %d x %d\n", blockPDF.x, blockPDF.y,
         gridPDF.x, gridPDF.y);
  printf("design: %d x %d\n", design.data_per_block, design.params_per_block);

  printf("nparams: %d\n", params.rows);
#endif

  mvnpdf_k<<<gridPDF,blockPDF,sharedMemSize>>>(data, params, design, d_pdf);
  return cudaSuccess;
}
