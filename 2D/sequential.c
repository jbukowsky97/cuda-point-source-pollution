#include <stdio.h>
#include <stdlib.h>

#define NUM_ARGS 9

unsigned long long int convertToLLU(char* str, char* inputName, int compare) {
    const unsigned long long int temp = strtoull(str, NULL, 0);
    if (temp < compare) {
        printf("%s must be greater than or equal to %d\n", inputName, compare);
        exit(EXIT_FAILURE);
    }
    return temp;
}

double convertToDouble(char* str, char* inputName, int compare) {
    const double temp = atof(str);
    if (temp <= compare) {
        printf("%s must be greater than %d\n", inputName, compare);
        exit(EXIT_FAILURE);
    }
    return temp;
}

unsigned long long int cIndex(const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    return slicesRow * row + col;
}

double cLeft(double* grid, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (col == 0) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row, col - 1)];
}

double cRight(double* grid, const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int row, const unsigned long long int col) {
    if (col == slicesCol - 1) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row, col + 1)];
}

double cUp(double* grid, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == 0) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row - 1, col)];
}

double cDown(double* grid, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == slicesRow - 1) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row + 1, col)];
}

int main(int argc, char** argv) {
    if (argc < NUM_ARGS || argc % 3 != 0) {
        printf("Usage:\n\t%s <grid rows> <grid columns> <total time> <desired row> <desired column> [<impulse 1 row> <impulse 1 column> <impulse 1 concentration>...<impulse N row> <impulse N column> <impulse N concentration>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int slicesRow = convertToLLU(argv[1], "Grid Rows", 1);
    const unsigned long long int slicesCol = convertToLLU(argv[2], "Grid Columns", 1);
    const unsigned long long int totalTime = convertToLLU(argv[3], "Total Time", 0);
    const unsigned long long int desiredPointRow = convertToLLU(argv[4], "Desired Row", 0);
    const unsigned long long int desiredPointCol = convertToLLU(argv[5], "Desired Column", 0);

    if (desiredPointRow > slicesRow || desiredPointCol > slicesCol) {
        printf("Desired point must be within grid\n");
        exit(EXIT_FAILURE);
    }

    const int numImpulses = (argc - NUM_ARGS) / 3 + 1;
    unsigned long long int* impulseLocations = malloc(2 * numImpulses * sizeof(unsigned long long int));
    double* concentrations = malloc(numImpulses * sizeof(double));
    int arrayIndex = 0;
    int argIndex = NUM_ARGS - 3;
    while (arrayIndex < numImpulses) {
        impulseLocations[arrayIndex * 2] = convertToLLU(argv[argIndex++], "Impulse Row", 0);
        impulseLocations[arrayIndex * 2 + 1] = convertToLLU(argv[argIndex++], "Impulse Column", 0);
        if (impulseLocations[arrayIndex * 2] >= slicesRow || impulseLocations[arrayIndex * 2 + 1] >= slicesCol) {
            printf("Impulse Location must be within grid\n");
            free(impulseLocations);
            free(concentrations);
            exit(EXIT_FAILURE);
        }
        concentrations[arrayIndex] = convertToDouble(argv[argIndex++], "Impulse Concentration", 0);
        arrayIndex++;
    }

    double* oldGrid = (double*) calloc(slicesRow * slicesCol, sizeof(double));
    double* newGrid = (double*) malloc(slicesRow * slicesCol * sizeof(double));
    double* temp;

    if (oldGrid == NULL || newGrid == NULL) {
        printf("Error while allocating memory\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numImpulses; i++) {
        oldGrid[cIndex(slicesRow, impulseLocations[i * 2], impulseLocations[i * 2 + 1])] = concentrations[i];
    }

    free(impulseLocations);
    free(concentrations);

    double total;
    for (unsigned long long int i = 0; i < totalTime; i++) {
        for (unsigned long long int row = 0; row < slicesRow; row++) {
            for (unsigned long long int col = 0; col < slicesCol; col++) {
                total = 0.0;
                total += cLeft(oldGrid, slicesRow, row, col);
                total += cRight(oldGrid, slicesRow, slicesCol, row, col);
                total += cUp(oldGrid, slicesRow, row, col);
                total += cDown(oldGrid, slicesRow, row, col);
                newGrid[cIndex(slicesRow, row, col)] = total / 4.0;
            }
        }
        temp = oldGrid;
        oldGrid = newGrid;
        newGrid = temp;
    }

    printf("%f\n", oldGrid[cIndex(slicesRow, desiredPointRow, desiredPointCol)]);

    free(oldGrid);
    free(newGrid);
}