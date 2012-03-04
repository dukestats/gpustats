// Exercise 1 from http://webapp.dam.brown.edu/wiki/SciComp/CudaExercises

// Transposition of a matrix
// by Hendrik Riedmann <riedmann@dam.brown.edu>
// Andrew Cron added bounds checks ...

#define BLOCK_SIZE %(block_size)d
    #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
    #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

    __global__ void transpose(float *A_t, float *A, int a_width, int a_height)
    {
        // Base indices in A and A_t
        int base_idx_a   = blockIdx.x * BLOCK_SIZE + 
	blockIdx.y * A_BLOCK_STRIDE;
        int base_idx_a_t = blockIdx.y * BLOCK_SIZE + 
	blockIdx.x * A_T_BLOCK_STRIDE;

        // Global indices in A and A_t
        int glob_idx_a = base_idx_a + threadIdx.x + a_width * threadIdx.y;
	int y_pos = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int x_pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int glob_idx_a_t = base_idx_a_t + threadIdx.x + a_height * threadIdx.y;

        __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE+1];

	if( x_pos < a_width && y_pos < a_height ){
            // Store transposed submatrix to shared memory
            A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a];
          
          __syncthreads();

          // Write transposed submatrix to global memory
          A_t[glob_idx_a_t] = A_shared[threadIdx.x][threadIdx.y];
	}

    }



