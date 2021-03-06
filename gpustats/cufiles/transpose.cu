// Exercise 1 from http://webapp.dam.brown.edu/wiki/SciComp/CudaExercises

// Transposition of a matrix
// by Hendrik Riedmann <riedmann@dam.brown.edu>
// Andrew Cron added bounds checks ...

// Andrew Cron added Z grid dimension to X for larger matrices

#define BLOCK_SIZE %(block_size)d
    #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
    #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

    __global__ void transpose(float *A_t, float *A, int a_width, int a_height)
    {
	int bidx = blockIdx.x + blockIdx.z;
        // Base indices in A and A_t
        int base_idx_a   = bidx * BLOCK_SIZE + 
	blockIdx.y * A_BLOCK_STRIDE;
        int base_idx_a_t = blockIdx.y * BLOCK_SIZE + 
	bidx * A_T_BLOCK_STRIDE;

        // Global indices in A and A_t
        int glob_idx_a = base_idx_a + threadIdx.x + a_width * threadIdx.y;
        int glob_idx_a_t = base_idx_a_t + threadIdx.x + a_height * threadIdx.y;

	int a_x_pos = bidx * BLOCK_SIZE + threadIdx.x;
	int a_y_pos = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int at_x_pos = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	int at_y_pos = bidx * BLOCK_SIZE + threadIdx.y;

        __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE+1];

	if( a_x_pos < a_width && a_y_pos < a_height ){
            // Store transposed submatrix to shared memory
            A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a];
        }          
        __syncthreads();
        if( at_x_pos < a_height && at_y_pos < a_width ){
            // Write transposed submatrix to global memory
            A_t[glob_idx_a_t] = A_shared[threadIdx.x][threadIdx.y];
	}

    }



