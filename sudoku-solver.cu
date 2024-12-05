#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>

#define GRID_SIZE 81
#define THREADS_PER_BLOCK 1

// Helper function to check if placing a number is valid
__device__ bool is_valid(int *board, int row, int col, int num) {
    // Check the row
    for (int i = 0; i < 9; i++) {
        if (board[row * 9 + i] == num) {
            return false;
        }
    }

    // Check the column
    for (int i = 0; i < 9; i++) {
        if (board[i * 9 + col] == num) {
            return false;
        }
    }

    // Check the 3x3 sub-grid
    int box_row_start = (row / 3) * 3;
    int box_col_start = (col / 3) * 3;
    for (int i = box_row_start; i < box_row_start + 3; i++) {
        for (int j = box_col_start; j < box_col_start + 3; j++) {
            if (board[i * 9 + j] == num) {
                return false;
            }
        }
    }

    return true;
}

// Device function to find the next empty cell using MRV heuristic
// Returns true if an MRV cell is found.
// Returns false if no empty cells (puzzle solved) or no suitable cell (no solution).
// If it returns false and sets row=-1,col=-1, it means no solution from this configuration.
__device__ bool find_empty(int *board, int *row, int *col) {
    int best_domain_size = 10; // domain size can't exceed 9, start with invalid large number
    int best_row = -1;
    int best_col = -1;
    bool found_empty = false;
    bool all_zero_domains = true;

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i * 9 + j] == 0) {
                found_empty = true;
                // Count domain size
                int domain_size = 0;
                for (int num = 1; num <= 9; num++) {
                    if (is_valid(board, i, j, num)) {
                        domain_size++;
                    }
                }
                if (domain_size > 0) {
                    all_zero_domains = false;
                    if (domain_size < best_domain_size) {
                        best_domain_size = domain_size;
                        best_row = i;
                        best_col = j;
                    }
                }
            }
        }
    }

    if (!found_empty) {
        // No empty cell means puzzle is solved
        return false;
    }

    if (all_zero_domains) {
        // Every empty cell has zero possibilities => no solution from this configuration
        *row = -1;
        *col = -1;
        return false;
    }

    // We found a suitable MRV cell
    *row = best_row;
    *col = best_col;
    return true;
}

// Explicit backtracking implementation for solving Sudoku
__device__ bool solve(int *board) {
    int stack[GRID_SIZE][2];
    int top = -1;
    int row, col;

    if (!find_empty(board, &row, &col)) {
        // If no cell found and row,col != (-1,-1), puzzle is solved
        if (row == -1 && col == -1) {
            // row = -1 and col = -1 means no solution from this config
            return false;
        } else {
            return true; // solved
        }
    }

    // We have an MRV cell at (row, col)
    stack[++top][0] = row;
    stack[top][1] = col;

    while (top >= 0) {
        row = stack[top][0];
        col = stack[top][1];

        bool placed = false;
        int start_val = board[row * 9 + col]; 
        for (int num = start_val + 1; num <= 9; num++) {
            if (is_valid(board, row, col, num)) {
                board[row * 9 + col] = num;
                placed = true;
                break;
            }
        }

        if (placed) {
            if (!find_empty(board, &row, &col)) {
                // If no cell found
                if (row == -1 && col == -1) {
                    // no solution from this branch
                    // restore this cell and backtrack
                    board[stack[top][0] * 9 + stack[top][1]] = 0;
                    top--;
                } else {
                    // puzzle solved
                    return true;
                }
            } else {
                stack[++top][0] = row;
                stack[top][1] = col;
            }
        } else {
            // Reset this cell and backtrack
            board[stack[top][0] * 9 + stack[top][1]] = 0;
            top--;
        }
    }
    return false; // unsolvable
}

// Kernel for solving multiple Sudoku puzzles in parallel
__global__ void solve_sudokus(int *boards, int num_boards) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_boards) {
        int *board = boards + idx * GRID_SIZE;
        if (solve(board)) {
            printf("Puzzle %d solved successfully.\n", idx);
        } else {
            printf("Puzzle %d is unsolvable.\n", idx);
        }
    }
}

// Host function for printing a Sudoku board
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

// Host code for managing CUDA memory and invoking the kernel
int main() {
    const int num_boards = 2;
    int boards[num_boards][GRID_SIZE] = {
        {9, 0, 0, 0, 3, 5, 0, 0, 0, 0, 0, 1, 4, 8, 0, 0, 5, 9, 3, 4, 0, 0, 0, 6, 2, 1, 0, 4, 0, 6, 5, 1, 0, 8, 3, 2, 0, 2, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 6, 2, 8, 0, 0, 1, 0, 0, 0, 0, 0, 7, 0, 0, 4, 2, 0, 0, 9, 0, 0, 5, 8, 0, 0, 0, 0, 0, 4, 1, 9, 0, 0},
        // ... add other puzzles ...
        {4, 0, 0, 0, 0, 0, 8, 0, 5, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 8, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, 7, 0, 5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0}
    };

    int *d_boards;
    size_t size = num_boards * GRID_SIZE * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_boards, size);

    // Copy boards to device memory
    cudaMemcpy(d_boards, boards, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    solve_sudokus<<<num_boards, THREADS_PER_BLOCK>>>(d_boards, num_boards);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(boards, d_boards, size, cudaMemcpyDeviceToHost);

    // Print the solved boards
    for(int i = 0; i < num_boards; i++) {
        printf("Solved Board %d:\n", i);
        print_board(boards[i]);
        printf("\n");
    }

    // Free device memory
    cudaFree(d_boards);

    return 0;
}