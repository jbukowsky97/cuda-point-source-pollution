#include <stdio.h>
#include <stdlib.h>

#define NUM_ARGS 6

double gpuCalculate(const unsigned long long int numSlices, const unsigned long long int totalTime, const unsigned long long int desiredPoint, const int numImpulses, const unsigned long long int* impulses, const double* concentrations);

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

int main(int argc, char** argv) {
    if (argc < NUM_ARGS || argc % 2 != 0) {
        printf("Usage:\n\t%s <cylinder slices> <total time> <desired point> <impulse 1 location> <impulse 1 concentration>...<impulse N location> <impulse N concentration> \n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int numSlices = convertToLLU(argv[1], "Cylinder Slices", 1);
    const unsigned long long int totalTime = convertToLLU(argv[2], "Total Time", 0);
    const unsigned long long int desiredPoint = convertToLLU(argv[3], "Desired Point", 0);

    if (desiredPoint > numSlices) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }

    const int numImpulses = (argc - NUM_ARGS) / 2 + 1;
    unsigned long long int* impulseLocations = malloc(numImpulses * sizeof(unsigned long long int));
    double* concentrations = malloc(numImpulses * sizeof(double));
    int arrayIndex = 0;
    int argIndex = NUM_ARGS - 2;
    while (arrayIndex < numImpulses) {
        impulseLocations[arrayIndex] = convertToLLU(argv[argIndex++], "Impulse Location", 0);
        if (impulseLocations[arrayIndex] >= numSlices) {
            printf("Impulse Location must be within cylinder\n");
            free(impulseLocations);
            free(concentrations);
            exit(EXIT_FAILURE);
        }
        concentrations[arrayIndex] = convertToDouble(argv[argIndex++], "Impulse Concentration", 0);
        arrayIndex++;
    }

    double answer = gpuCalculate(numSlices, totalTime, desiredPoint, numImpulses, impulseLocations, concentrations);
    printf("%f\n", answer);

    free(impulseLocations);
    free(concentrations);
}