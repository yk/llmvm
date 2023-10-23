#include <stdio.h>

// very simple ascii snake game on 4x4 grid

#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 4

#define GRID_ACCESS(grid, row, col) grid[row * COLS + col]

int main() {
    // Define the grid
    char grid[ROWS * COLS];

    // Initialize the random number generator
    srand(42);

    // Initialize the grid with empty spaces
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            GRID_ACCESS(grid, i, j) = ' ';
        }
    }

    // Initialize the snake at a random position
    int snake_row = rand() % ROWS;
    int snake_col = rand() % COLS;
    GRID_ACCESS(grid, snake_row, snake_col) = 'o';

    // Place the food at a random location
    int food_row = rand() % ROWS;
    int food_col = rand() % COLS;
    GRID_ACCESS(grid, food_row, food_col) = '*';

    // Print the initial grid
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%c ", GRID_ACCESS(grid, i, j));
        }
        printf("\n");
    }



    // Game loop
    while (1) {
        // Take user input for the direction of the snake
        char direction;
        printf("Enter direction (w/a/s/d): ");
        scanf(" %c", &direction);

        // Erase the snake from the grid
        GRID_ACCESS(grid, snake_row, snake_col) = ' ';

        // Update the position of the snake based on the user input
        switch (direction) {
            case 'w':
                snake_row--;
                break;
            case 's':
                snake_row++;
                break;
            case 'a':
                snake_col--;
                break;
            case 'd':
                snake_col++;
                break;
        }

        // Check if the snake has collided with the wall or itself
        if (snake_row < 0 || snake_row >= ROWS || snake_col < 0 || snake_col >= COLS || GRID_ACCESS(grid, snake_row, snake_col) == 'o') {
            printf("Game over!\n");
            break;
        }

        // Check if the snake has collided with the food
        if (snake_row == food_row && snake_col == food_col) {
            printf("You won!\n");
            break;
        }

        // Update the grid with the new position of the snake
        GRID_ACCESS(grid, snake_row, snake_col) = 'o';


        // Print the updated grid
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                printf("%c ", GRID_ACCESS(grid, i, j));
            }
            printf("\n");
        }

    }

    return 0;
}
