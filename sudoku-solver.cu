#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

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

#define NUM_BOARDS 95

#include <stdio.h>
#include <stdlib.h>

#define NUM_BOARDS 95
#define GRID_SIZE 81

void read_sudoku_boards(const char *filename, int boards[NUM_BOARDS][GRID_SIZE]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file '%s'\n", filename);
        exit(1);
    }

    int board_index = 0;
    int cell_index = 0;
    char line[128];

    while (fgets(line, sizeof(line), file)) {
        // Skip empty lines
        if (line[0] == '\n' || line[0] == '\0') {
            continue;
        }

        // Parse 9 characters from the line and add to the current board
        for (int i = 0; i < 9; i++) {
            if (line[i] >= '0' && line[i] <= '9') {
                boards[board_index][cell_index++] = line[i] - '0';
            }
        }

        // If a board is fully read, move to the next board
        if (cell_index == GRID_SIZE) {
            board_index++;
            cell_index = 0;

            if (board_index > NUM_BOARDS) {
                fprintf(stderr, "Error: More boards in file than expected (%d).\n", NUM_BOARDS);
                fclose(file);
                exit(1);
            }
        }
    }

    fclose(file);

    // Check if exactly the expected number of boards was read
    if (board_index != NUM_BOARDS) {
        fprintf(stderr, "Error: Fewer boards in file than expected (%d).\n", NUM_BOARDS);
        exit(1);
    }
}

void save_sudoku_boards(const char *filename, int boards[NUM_BOARDS][GRID_SIZE]) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file '%s' for writing\n", filename);
        return;
    }

    for (int b = 0; b < NUM_BOARDS; b++) {
        fprintf(file, "|-----------------------|\n");

        for (int row = 0; row < 9; row++) {
            fprintf(file, "|");
            for (int col = 0; col < 9; col++) {
                int idx = row * 9 + col;
                fprintf(file, "%2d", boards[b][idx]);
                if (col % 3 == 2) {
                    fprintf(file, " |");
                }
            }
            fprintf(file, "\n");
            if (row % 3 == 2 && row != 8) {
                fprintf(file, "|-----------------------|\n");
            }
        }

        fprintf(file, "|-----------------------|\n");
        if (b < NUM_BOARDS - 1) {
            fprintf(file, "\n");
        }
    }

    fclose(file);
}

// Host code for managing CUDA memory and invoking the kernel
int main() {
    const int num_boards = 95;

    // Statically declared array for NUM_BOARDS boards
    int boards[NUM_BOARDS][GRID_SIZE];

    // Read boards from file
    read_sudoku_boards("inp.in", boards);

    // Print the solved boards
    // for(int i = 0; i < num_boards; i++) {
    //     printf("Unsolved Board %d:\n", i);
    //     print_board(boards[i]);
    // }

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
    // for(int i = 0; i < num_boards; i++) {
    //     printf("Solved Board %d:\n", i);
    //     print_board(boards[i]);
    // }

    // Save boards to file
    save_sudoku_boards("sol.out", boards);

    // Free device memory
    cudaFree(d_boards);

    return 0;
}