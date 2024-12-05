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

// Device function to find the next empty cell
__device__ bool find_empty(int *board, int *row, int *col) {
    int min_domain_size = 10; // Start with a value greater than the max possible domain size (9)
    int best_row = -1;
    int best_col = -1;

    // Iterate over all cells
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            int val = board[i * 9 + j];
            if (val == 0) {
                // Empty cell, count possibilities
                int domain_size = 0;
                for (int num = 1; num <= 9; num++) {
                    // Check if placing 'num' is valid
                    bool can_place = true;
                    // Row check
                    for (int x = 0; x < 9; x++) {
                        if (board[i * 9 + x] == num) {
                            can_place = false;
                            break;
                        }
                    }
                    if (!can_place) continue;

                    // Column check
                    for (int x = 0; x < 9; x++) {
                        if (board[x * 9 + j] == num) {
                            can_place = false;
                            break;
                        }
                    }
                    if (!can_place) continue;

                    // 3x3 box check
                    int box_row_start = (i / 3) * 3;
                    int box_col_start = (j / 3) * 3;
                    for (int rr = box_row_start; rr < box_row_start + 3 && can_place; rr++) {
                        for (int cc = box_col_start; cc < box_col_start + 3; cc++) {
                            if (board[rr * 9 + cc] == num) {
                                can_place = false;
                                break;
                            }
                        }
                    }

                    if (can_place) domain_size++;
                    if (domain_size > 1 && domain_size >= min_domain_size) {
                        // If domain_size already exceeds current min_domain_size (or is >1 and min_domain_size=1)
                        // no need to check further for this cell
                        break;
                    }
                }

                // If domain_size < min_domain_size, update
                if (domain_size < min_domain_size) {
                    min_domain_size = domain_size;
                    best_row = i;
                    best_col = j;
                    // If domain_size == 1, return immediately
                    if (domain_size == 1) {
                        *row = best_row;
                        *col = best_col;
                        return true;
                    }
                }
            }
        }
    }

    // If no empty cells found, puzzle is solved
    if (best_row == -1 && best_col == -1) {
        return false; 
    }

    // Return the cell with minimal domain size found
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
        return true; // No empty cells, puzzle solved
    }

    stack[++top][0] = row;
    stack[top][1] = col;

    while (top >= 0) {
        row = stack[top][0];
        col = stack[top][1];

        bool placed = false;
        for (int num = board[row * 9 + col] + 1; num <= 9; num++) {
            if (is_valid(board, row, col, num)) {
                board[row * 9 + col] = num;
                placed = true;
                break;
            }
        }

        if (placed) {
            if (find_empty(board, &row, &col)) {
                stack[++top][0] = row;
                stack[top][1] = col;
            } else {
                return true; // Solved
            }
        } else {
            board[stack[top][0] * 9 + stack[top][1]] = 0; // Reset cell
            top--; // Backtrack
        }
    }

    return false; // Unsolvable
}

// Host function to check if board is valid
__device__ bool is_board_valid(int *board) {
    // Check rows
    for (int r = 0; r < 9; r++) {
        int seen[10] = {0}; // track digits 1-9
        for (int c = 0; c < 9; c++) {
            int val = board[r*9 + c];
            if (val != 0) {
                if (seen[val]) return false;
                seen[val] = 1;
            }
        }
    }

    // Check columns
    for (int c = 0; c < 9; c++) {
        int seen[10] = {0};
        for (int r = 0; r < 9; r++) {
            int val = board[r*9 + c];
            if (val != 0) {
                if (seen[val]) return false;
                seen[val] = 1;
            }
        }
    }

    // Check 3x3 sub-grids
    for (int br = 0; br < 3; br++) {
        for (int bc = 0; bc < 3; bc++) {
            int seen[10] = {0};
            for (int r = br*3; r < br*3+3; r++) {
                for (int c = bc*3; c < bc*3+3; c++) {
                    int val = board[r*9 + c];
                    if (val != 0) {
                        if (seen[val]) return false;
                        seen[val] = 1;
                    }
                }
            }
        }
    }

    // If no violations found
    return true;
}

// Kernel for solving multiple Sudoku puzzles in parallel
__global__ void solve_sudokus(int *boards, int num_boards) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_boards) {
        int *board = boards + idx * GRID_SIZE;
        if(is_board_valid(board)) {
            if (solve(board)) {
                printf("Puzzle %d solved successfully.\n", idx);
            } else {
                printf("Puzzle %d is unsolvable.\n", idx);
            }
        }
        else {
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
    const int num_boards = 14;
    int boards[num_boards][GRID_SIZE] = {
        {9, 0, 0, 0, 3, 5, 0, 0, 0, 0, 0, 1, 4, 8, 0, 0, 5, 9, 3, 4, 0, 0, 0, 6, 2, 1, 0, 4, 0, 6, 5, 1, 0, 8, 3, 2, 0, 2, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 6, 2, 8, 0, 0, 1, 0, 0, 0, 0, 0, 7, 0, 0, 4, 2, 0, 0, 9, 0, 0, 5, 8, 0, 0, 0, 0, 0, 4, 1, 9, 0, 0},
        {0, 7, 0, 0, 0, 2, 5, 0, 9, 5, 8, 0, 3, 4, 0, 0, 0, 0, 2, 0, 1, 5, 0, 9, 0, 0, 8, 1, 0, 3, 0, 0, 0, 0, 5, 0, 9, 5, 6, 0, 3, 0, 0, 7, 1, 7, 2, 8, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 0, 0, 0, 0, 0, 6, 0, 5, 3, 1, 5, 4, 6, 0, 0, 0, 2},
        {8, 9, 3, 1, 4, 0, 0, 0, 0, 4, 2, 0, 3, 7, 5, 8, 1, 0, 1, 5, 0, 0, 9, 0, 2, 0, 0, 2, 0, 0, 0, 6, 7, 0, 9, 8, 0, 0, 0, 0, 3, 1, 0, 0, 0, 3, 8, 0, 5, 2, 9, 0, 7, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 1, 0, 0},
        {0, 7, 0, 1, 0, 2, 0, 6, 0, 2, 0, 0, 5, 0, 0, 0, 3, 9, 0, 5, 0, 9, 0, 0, 1, 4, 0, 0, 3, 0, 4, 0, 5, 6, 8, 0, 0, 8, 5, 0, 7, 1, 0, 9, 0, 0, 0, 0, 3, 0, 0, 4, 5, 0, 7, 6, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 7, 0, 3, 8, 1, 6, 0, 9, 0, 2, 5, 0, 3, 7, 0},
        {7, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 3, 0, 0, 7, 2, 0, 9, 4, 0, 6, 0, 0, 1, 0, 0, 0, 5, 4, 0, 9, 1, 0, 0, 6, 0, 0, 0, 8, 7, 0, 3, 0, 0, 0, 7, 1, 5, 3, 6, 0, 4, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 0, 6, 9, 5, 9, 0, 4, 6, 8, 2, 0, 0},
        {0, 7, 1, 0, 3, 0, 0, 9, 6, 0, 0, 3, 0, 6, 0, 0, 0, 5, 6, 5, 0, 7, 8, 9, 0, 0, 3, 2, 0, 8, 0, 0, 0, 0, 0, 7, 1, 0, 5, 8, 7, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 4, 0, 0, 0, 6, 0, 4, 0, 2, 3, 9, 7, 3, 2, 5, 0, 0, 6, 0, 4, 4, 1, 9, 0, 2, 0, 0, 7, 8},
        {6, 0, 0, 3, 5, 7, 8, 9, 4, 0, 0, 0, 1, 2, 0, 6, 0, 0, 0, 0, 8, 4, 0, 0, 7, 0, 0, 0, 0, 0, 0, 4, 1, 9, 8, 6, 1, 0, 0, 9, 0, 0, 0, 7, 3, 8, 9, 0, 0, 0, 0, 4, 5, 0, 0, 0, 5, 8, 7, 0, 1, 0, 9, 7, 0, 0, 5, 1, 9, 0, 0, 8, 0, 0, 1, 6, 3, 0, 0, 0, 7},
        {0, 1, 0, 2, 4, 3, 0, 9, 7, 0, 0, 0, 8, 0, 9, 2, 0, 0, 0, 9, 0, 7, 6, 5, 4, 1, 0, 1, 6, 2, 0, 0, 0, 9, 3, 0, 0, 0, 0, 0, 0, 6, 0, 0, 1, 9, 0, 0, 0, 0, 4, 5, 8, 6, 3, 2, 0, 4, 5, 7, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 3, 7, 0},
        {5, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 8, 5, 6, 3, 0, 0, 0, 0, 6, 4, 0, 0, 0, 0, 0, 0, 2, 7, 6, 4, 1, 0, 0, 0, 1, 0, 5, 0, 0, 0, 2, 4, 3, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0, 0, 8, 0, 0, 5, 0, 3, 7, 0, 4, 0, 3, 0, 8, 5, 9, 1, 0, 0, 3, 9, 1, 4, 0, 0, 2},
        {6, 0, 8, 1, 3, 0, 5, 9, 0, 0, 9, 0, 0, 5, 0, 0, 1, 0, 0, 4, 5, 8, 7, 9, 0, 3, 6, 4, 0, 0, 0, 0, 1, 7, 5, 0, 2, 0, 1, 6, 0, 5, 0, 0, 0, 5, 3, 9, 0, 2, 0, 0, 4, 0, 9, 0, 3, 0, 0, 0, 0, 0, 5, 0, 1, 0, 0, 0, 2, 9, 7, 3, 0, 0, 2, 0, 1, 3, 0, 6, 0},
        {0, 6, 0, 0, 2, 0, 0, 0, 5, 3, 0, 0, 0, 0, 8, 6, 0, 0, 0, 9, 7, 0, 5, 0, 0, 0, 0, 0, 0, 0, 2, 9, 5, 8, 0, 1, 0, 8, 0, 0, 0, 3, 0, 9, 0, 0, 0, 3, 0, 0, 0, 4, 5, 0, 0, 2, 0, 0, 0, 1, 9, 4, 0, 7, 0, 0, 5, 0, 0, 0, 0, 8, 0, 4, 1, 0, 3, 0, 5, 0, 7},
        {3, 8, 0, 0, 0, 0, 7, 6, 0, 0, 1, 2, 6, 0, 0, 0, 8, 4, 7, 0, 0, 0, 0, 9, 1, 0, 0, 0, 0, 0, 0, 9, 7, 0, 3, 0, 8, 0, 0, 5, 4, 0, 9, 1, 0, 0, 6, 9, 1, 8, 3, 5, 0, 7, 0, 0, 8, 0, 0, 1, 6, 0, 0, 0, 7, 0, 9, 0, 8, 0, 0, 0, 5, 9, 0, 3, 0, 4, 2, 7, 0},
        {0, 0, 6, 1, 5, 0, 0, 0, 8, 0, 7, 3, 0, 0, 8, 5, 2, 9, 0, 0, 0, 0, 7, 0, 0, 1, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 6, 4, 0, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 1, 0, 0, 7, 5, 0, 6, 1, 0, 8, 0, 0, 9, 6, 1, 0, 8, 0, 7, 4, 0},
        {4, 0, 0, 0, 0, 0, 6, 0, 5, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 8, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, 7, 0, 5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0}
    };


    int *d_boards;
    size_t size = num_boards * GRID_SIZE * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_boards, size);

    // Copy boards to device memory
    cudaMemcpy(d_boards, boards, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    solve_sudokus<<<num_boards, THREADS_PER_BLOCK>>>(d_boards, num_boards);

    // Copy results back to host
    cudaMemcpy(boards, d_boards, size, cudaMemcpyDeviceToHost);

    // Print the solved boards
    for(int i = 0; i < num_boards; i++) {
        printf("Solved Board %d:\n", i);
        print_board(boards[i]);
    }

    // Free device memory
    cudaFree(d_boards);

    return 0;
}