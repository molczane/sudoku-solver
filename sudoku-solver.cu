#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>

#define GRID_SIZE 81
#define THREADS_PER_BLOCK 81 // one thread per cell

__device__ bool parallel_is_valid(int *board, int row, int col, int num) {
    __shared__ bool conflict;
    // Initialize
    if (threadIdx.x == 0) conflict = false;
    __syncthreads();

    int tid = threadIdx.x;

    // The puzzle:
    // Row check: threads 0..8 each check one cell in the row
    // Col check: threads 9..17 check one cell in the column
    // Box check: threads 18..26 check one cell in the box
    // The rest of the threads do nothing for simplicity

    // Row check
    if (tid < 9) {
        int val = board[row * 9 + tid];
        if (val == num) {
            atomicExch((int*)&conflict, true);
        }
    }

    // Column check
    if (tid >= 9 && tid < 18) {
        int c_tid = tid - 9;
        int val = board[c_tid * 9 + col];
        if (val == num) {
            atomicExch((int*)&conflict, true);
        }
    }

    // Box check
    if (tid >= 18 && tid < 27) {
        int b_tid = tid - 18;
        int box_row_start = (row / 3) * 3;
        int box_col_start = (col / 3) * 3;
        int r = box_row_start + b_tid / 3;
        int c = box_col_start + b_tid % 3;
        int val = board[r * 9 + c];
        if (val == num) {
            atomicExch((int*)&conflict, true);
        }
    }

    __syncthreads();
    // Only thread 0 returns the result
    bool result = true;
    if (threadIdx.x == 0) {
        result = !conflict;
    }
    __syncthreads();
    return result;
}

// Device function to find the next empty cell
// Only thread 0 will do this, others just wait
__device__ bool find_empty(int *board, int *row, int *col) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i * 9 + j] == 0) {
                    *row = i;
                    *col = j;
                    return true;
                }
            }
        }
    }
    __syncthreads();
    // If thread 0 didn't find anything, no empty cell
    if (threadIdx.x == 0) return false;
    // Non-zero threads just return after sync
    return false;
}

// Explicit backtracking implementation for solving Sudoku
// Only thread 0 will run the backtracking logic, others are helpers for is_valid checks
__device__ bool solve(int *board) {
    __shared__ int stack[GRID_SIZE][2];
    __shared__ int top;
    if (threadIdx.x == 0) top = -1;
    __syncthreads();

    int row, col;

    if (threadIdx.x == 0) {
        if (!find_empty(board, &row, &col)) {
            // no empty cell => solved
        }
    }
    __syncthreads();

    // If no empty cell found by thread 0
    bool no_empty;
    if (threadIdx.x == 0) no_empty = !find_empty(board, &row, &col);
    __syncthreads();

    if (no_empty && threadIdx.x == 0) return true; // solved

    // If empty cell found, push on stack
    if (threadIdx.x == 0) {
        stack[++top][0] = row;
        stack[top][1] = col;
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            if (top < 0) {
                // No solution
            }
        }
        __syncthreads();

        if (threadIdx.x == 0 && top < 0) return false; // no solution

        if (threadIdx.x == 0) {
            row = stack[top][0];
            col = stack[top][1];

            bool placed = false;
            int start_val = board[row * 9 + col];
            for (int num = start_val + 1; num <= 9; num++) {
                __syncthreads(); // ensure all threads ready
                bool valid = parallel_is_valid(board, row, col, num);
                __syncthreads();
                if (threadIdx.x == 0 && valid) {
                    board[row * 9 + col] = num;
                    placed = true;
                    break;
                }
            }

            if (placed) {
                // find next empty cell
                if (!find_empty(board, &row, &col)) {
                    // no empty => solved
                    // break loop
                    top = -999; // special code to indicate done
                } else {
                    stack[++top][0] = row;
                    stack[top][1] = col;
                }
            } else {
                // reset cell and backtrack
                board[row * 9 + col] = 0;
                top--;
            }
        }

        __syncthreads();
        if (threadIdx.x == 0 && top == -999) return true; // solved
        __syncthreads();
    }

    // unreachable
    return false;
}

// Kernel for solving multiple Sudoku puzzles in parallel
__global__ void solve_sudokus(int *boards, int num_boards) {
    int puzzle_idx = blockIdx.x;
    int *board = boards + puzzle_idx * GRID_SIZE;
    bool result = solve(board);
    if (threadIdx.x == 0) {
        if (result)
            printf("Puzzle %d solved successfully.\n", puzzle_idx);
        else
            printf("Puzzle %d is unsolvable.\n", puzzle_idx);
    }
}

void print_board(int *board) {
    for (int i = 0; i < 9; i++) {
        if (i % 3 == 0 && i != 0) {
            printf("---------------------\n");
        }
        for (int j = 0; j < 9; j++) {
            if (j % 3 == 0 && j != 0) {
                printf("| ");
            }
            printf("%d ", board[i * 9 + j]);
        }
        printf("\n");
    }
}

int main() {
    const int num_boards = 2;
    int boards[num_boards][GRID_SIZE] = {
        {9, 0, 0, 0, 3, 5, 0, 0, 0,
         0, 0, 1, 4, 8, 0, 0, 5, 9,
         3, 4, 0, 0, 0, 6, 2, 1, 0,
         4, 0, 6, 5, 1, 0, 8, 3, 2,
         0, 2, 0, 0, 0, 0, 6, 0, 0,
         0, 0, 0, 0, 0, 6, 2, 8, 0,
         0, 1, 0, 0, 0, 0, 0, 7, 0,
         0, 4, 2, 0, 0, 9, 0, 0, 5,
         8, 0, 0, 0, 0, 0, 4, 1, 9
        },
        // ... other puzzles ...
        {4, 0, 0, 0, 0, 0, 8, 0, 5,
         0, 3, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 7, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 2, 0, 0,
         0, 0, 0, 0, 0, 6, 0, 0, 0,
         0, 0, 0, 8, 0, 4, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 6, 0, 3, 0,
         0, 7, 0, 5, 0, 0, 2, 0, 0
        }
    };

    int *d_boards;
    size_t size = num_boards * GRID_SIZE * sizeof(int);
    cudaMalloc(&d_boards, size);
    cudaMemcpy(d_boards, boards, size, cudaMemcpyHostToDevice);

    // Launch one block per puzzle, 81 threads per block
    solve_sudokus<<<num_boards, THREADS_PER_BLOCK * 81>>>(d_boards, num_boards);
    cudaDeviceSynchronize();

    cudaMemcpy(boards, d_boards, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_boards; i++) {
        printf("Solved Board %d:\n", i);
        print_board(boards[i]);
        printf("\n");
    }

    cudaFree(d_boards);
    return 0;
}