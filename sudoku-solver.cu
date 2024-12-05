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
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i * 9 + j] == 0) {
                *row = i;
                *col = j;
                return true;
            }
        }
    }
    return false;
}

// Explicit backtracking implementation for solving Sudoku
__device__ bool solve(int *board) {
    int possible[GRID_SIZE]; // Bitmask for possible values of each cell
    int stack[GRID_SIZE][3];      // Stack for backtracking: [cell index, bitmask, last value tried]
    int top = -1;

    // Initialize possible values for each cell
    for (int i = 0; i < GRID_SIZE; i++) {
        if (board[i] == 0) {
            int row = i / 9;
            int col = i % 9;
            int subgrid = (row / 3) * 3 + (col / 3);

            // Start with all numbers (1–9) as possible
            possible[i] = 0x1FF;

            // Remove numbers already present in the row, column, or subgrid
            for (int j = 0; j < 9; j++) {
                int val_row = board[row * 9 + j];
                int val_col = board[j * 9 + col];
                int val_subgrid = board[(subgrid / 3) * 27 + (subgrid % 3) * 3 + (j / 3) * 9 + (j % 3)];
                if (val_row > 0) possible[i] &= ~(1 << (val_row - 1));
                if (val_col > 0) possible[i] &= ~(1 << (val_col - 1));
                if (val_subgrid > 0) possible[i] &= ~(1 << (val_subgrid - 1));
            }
        } else {
            possible[i] = 0; // Filled cells have no possibilities
        }
    }

    // Backtracking loop
    while (true) {
        // Find the cell with the least number of possible values (MRV heuristic)
        int min_index = -1;
        int min_count = 10; // More than the maximum possible (9)

        for (int i = 0; i < GRID_SIZE; i++) {
            if (board[i] == 0) {
                int count = __popc(possible[i]); // Count set bits in the bitmask
                if (count > 0 && count < min_count) {
                    min_count = count;
                    min_index = i;
                }
            }
        }

        // If no cell is left, the board is solved
        if (min_index == -1) return true;

        // Get the possible values for the selected cell
        int mask = possible[min_index];
        int row = min_index / 9;
        int col = min_index % 9;
        int subgrid = (row / 3) * 3 + (col / 3);

        // Push the cell onto the stack for backtracking
        stack[++top][0] = min_index;
        stack[top][1] = mask;
        stack[top][2] = 0; // Start with the first possible value

        while (top >= 0) {
            int cell = stack[top][0];
            mask = stack[top][1];
            int last_value = stack[top][2];

            // Find the next possible value for the cell
            int next_value = -1;
            for (int num = last_value + 1; num <= 9; num++) {
                if (mask & (1 << (num - 1))) {
                    next_value = num;
                    break;
                }
            }

            if (next_value == -1) {
                // No more possible values, backtrack
                board[cell] = 0;
                possible[cell] = stack[top--][1]; // Restore possibilities
            } else {
                // Place the value in the cell
                board[cell] = next_value;
                stack[top][2] = next_value;

                // Update constraints dynamically
                int bit = 1 << (next_value - 1);
                for (int j = 0; j < 9; j++) {
                    int row_cell = row * 9 + j;
                    int col_cell = j * 9 + col;
                    int subgrid_cell = (subgrid / 3) * 27 + (subgrid % 3) * 3 + (j / 3) * 9 + (j % 3);
                    possible[row_cell] &= ~bit;
                    possible[col_cell] &= ~bit;
                    possible[subgrid_cell] &= ~bit;
                }

                // Move to the next empty cell
                break;
            }
        }

        // If the stack is empty and no solution is found, the puzzle is unsolvable
        if (top < 0) return false;
    }
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
        for (int j = 0; j < 9; j++) {
            printf("%d ", board[i * 9 + j]);
        }
        printf("\n");
    }
}

// Host code for managing CUDA memory and invoking the kernel
int main() {
    const int num_boards = 13;
int boards[num_boards][GRID_SIZE] = {
    {9, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 1, 4, 8, 0, 0, 5, 9, 3, 4, 0, 0, 0, 6, 2, 1, 0, 4, 0, 6, 5, 1, 0, 8, 3, 2, 0, 2, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 6, 2, 8, 0, 0, 1, 0, 0, 0, 0, 0, 7, 0, 0, 4, 2, 0, 0, 9, 0, 0, 5, 8, 0, 0, 0, 0, 0, 4, 1, 9, 0, 0},
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
    {0, 0, 6, 1, 5, 0, 0, 0, 8, 0, 7, 3, 0, 0, 8, 5, 2, 9, 0, 0, 0, 0, 7, 0, 0, 1, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 6, 4, 0, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 1, 0, 0, 7, 5, 0, 6, 1, 0, 8, 0, 0, 9, 6, 1, 0, 8, 0, 7, 4, 0}
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
