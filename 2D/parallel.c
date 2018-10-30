#include <stdio.h>
#include <stdlib.h>

#define NUM_ARGS 9

double gpuCalculate(const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int totalTime, const unsigned long long int desiredPointRow, const unsigned long long int desiredPointCol, const int numImpulses, const unsigned long long int* impulses, const double* concentrations);

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

    double answer = gpuCalculate(slicesRow, slicesCol, totalTime, desiredPointRow, desiredPointCol, numImpulses, impulseLocations, concentrations);

    printf("%f\n", answer);

    free(impulseLocations);
    free(concentrations);
}