#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define GRID_SIZE 81
#define THREADS_PER_BLOCK 32

// Device function to validate constraints
__device__ bool is_valid(uint16_t possible) {
    return __popc(possible) == 1; // Only one possibility
}

__device__ uint16_t atomicOrCustom(uint16_t *address, uint16_t val) {
    unsigned int *base_address = (unsigned int *)((uintptr_t)address & ~2);
    unsigned int old = *base_address;
    unsigned int shift = ((uintptr_t)address & 2) * 8;
    uint16_t old_val;
    uint16_t new_val;

    do {
        old_val = (old >> shift) & 0xFFFF;
        new_val = old_val | val;
        old = atomicCAS(base_address, old, (old & ~(0xFFFF << shift)) | (new_val << shift));
    } while (old_val != new_val);

    return old_val;
}


// Optimized kernel for solving Sudoku
__global__ void solve_sudokus_optimized(uint32_t *boards, int num_boards) {
    int idx = blockIdx.x;
    if (idx >= num_boards) return;

    __shared__ uint32_t board[GRID_SIZE];
    __shared__ uint32_t row_constraints[9];
    __shared__ uint32_t col_constraints[9];
    __shared__ uint32_t subgrid_constraints[9];
    __shared__ int stack[GRID_SIZE][2]; // Stack for backtracking: [cell index, possibilities]
    __shared__ int stack_top;

    int thread_id = threadIdx.x;

    // Load board into shared memory
    for (int i = thread_id; i < GRID_SIZE; i += blockDim.x) {
        board[i] = boards[idx * GRID_SIZE + i];
    }
    __syncthreads();

    // Initialize constraints
    if (thread_id < 9) {
        row_constraints[thread_id] = 0;
        col_constraints[thread_id] = 0;
        subgrid_constraints[thread_id] = 0;
    }
    __syncthreads();

    // Populate initial constraints
    for (int i = thread_id; i < GRID_SIZE; i += blockDim.x) {
        int row = i / 9;
        int col = i % 9;
        int subgrid = (row / 3) * 3 + (col / 3);

        if (board[i] > 0) {
            uint32_t mask = 1 << (board[i] - 1);
            atomicOr(&row_constraints[row], mask);
            atomicOr(&col_constraints[col], mask);
            atomicOr(&subgrid_constraints[subgrid], mask);
        }
    }
    __syncthreads();

    // Initialize backtracking stack
    if (thread_id == 0) stack_top = -1;
    __syncthreads();

    bool progress = true;
    while (progress || stack_top >= 0) {
        progress = false;

        // Constraint propagation phase
        for (int i = thread_id; i < GRID_SIZE; i += blockDim.x) {
            if (board[i] == 0) { // Empty cell
                int row = i / 9;
                int col = i % 9;
                int subgrid = (row / 3) * 3 + (col / 3);

                uint32_t possible = ~(row_constraints[row] | col_constraints[col] | subgrid_constraints[subgrid]) & 0x1FF;

                if (__popc(possible) == 1) { // Deterministic cell
                    int value = __ffs(possible) - 1;
                    board[i] = value + 1;

                    uint32_t mask = 1 << value;
                    atomicOr(&row_constraints[row], mask);
                    atomicOr(&col_constraints[col], mask);
                    atomicOr(&subgrid_constraints[subgrid], mask);
                    progress = true;
                } else if (thread_id == 0 && possible != 0) {
                    // Push ambiguous cell to stack for backtracking
                    stack[++stack_top][0] = i;
                    stack[stack_top][1] = possible;
                }
            }
        }
        __syncthreads();

        // Backtracking phase
        if (thread_id == 0 && !progress && stack_top >= 0) {
            int cell = stack[stack_top][0];
            uint32_t possible = stack[stack_top][1];

            int row = cell / 9;
            int col = cell % 9;
            int subgrid = (row / 3) * 3 + (col / 3);

            // Try the next possibility
            int value = __ffs(possible) - 1; // Get the next possibility
            board[cell] = value + 1;

            uint32_t mask = 1 << value;
            atomicOr(&row_constraints[row], mask);
            atomicOr(&col_constraints[col], mask);
            atomicOr(&subgrid_constraints[subgrid], mask);

            // Update stack
            possible &= ~(1 << value); // Remove tried value
            if (possible == 0) {
                stack_top--; // Pop stack if no possibilities left
            } else {
                stack[stack_top][1] = possible; // Update possibilities
            }
            progress = true;
        }
        __syncthreads();
    }

    // Write the solved board back to global memory
    for (int i = thread_id; i < GRID_SIZE; i += blockDim.x) {
        boards[idx * GRID_SIZE + i] = board[i];
    }
}


// Host function to print a Sudoku board
void print_board(uint32_t *board) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            printf("%d ", board[i * 9 + j]);
        }
        printf("\n");
    }
}

// Host function
int main() {
    const int num_boards = 2; // Example number of puzzles
    uint32_t boards[num_boards][GRID_SIZE] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0, 6, 0, 0, 1, 9, 5, 0, 0, 0, 0, 9, 8, 0, 0, 0, 0, 6, 0, 8, 0, 0, 0, 6, 0, 0, 0, 3, 4, 0, 0, 8, 0, 3, 0, 0, 1, 7, 0, 0, 0, 2, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0, 2, 8, 0, 0, 0, 4, 1, 9, 0, 0, 5, 0, 0, 0, 0, 8, 0, 0, 7, 9},
        {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 0, 7, 0, 0, 9, 0, 2, 0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 4, 5, 7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 6, 8, 0, 0, 0, 8, 5, 0, 0, 0, 1, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0},
    };

    uint32_t *d_boards;
    size_t size = num_boards * GRID_SIZE * sizeof(uint32_t);

    // Print the solved boards
    for (int i = 0; i < num_boards; i++) {
        printf("Non-Solved Board %d:\n", i);
        print_board(boards[i]);
    }

    // Allocate device memory
    cudaMalloc(&d_boards, size);

    // Copy boards to device memory
    cudaMemcpy(d_boards, boards, size, cudaMemcpyHostToDevice);
    
    // Launch the kernel
    solve_sudokus_optimized<<<num_boards, THREADS_PER_BLOCK>>>(d_boards, num_boards);

    // Copy results back to host
    cudaMemcpy(boards, d_boards, size, cudaMemcpyDeviceToHost);

    // Print the solved boards
    for (int i = 0; i < num_boards; i++) {
        printf("Solved Board %d:\n", i);
        print_board(boards[i]);
    }

    // Free device memory
    cudaFree(d_boards);

    return 0;
}
