#include <stdio.h>
#include <stdlib.h>

unsigned long long int convertToNum(char* str) {
    const unsigned long long int temp = strtoull(str, NULL, 0);
    if (temp <= 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    return temp;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Usage:\n\t%s <cylinder slices> <total time> <point source concentration> <desired point>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int numSlices = convertToNum(argv[1]);
    const unsigned long long int totalTime = convertToNum(argv[2]);
    double concentration = atof(argv[3]);
    if (concentration < 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    const unsigned long long int desiredPoint = convertToNum(argv[4]);

    if (desiredPoint > numSlices) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }

    double* oldCylinder = (double*) calloc(numSlices, sizeof(double));
    double* newCylinder = (double*) malloc(numSlices * sizeof(double));
    double* temp;

    if (oldCylinder == NULL || newCylinder == NULL) {
        printf("Error while allocating memory\n");
        exit(EXIT_FAILURE);
    }

    oldCylinder[0] = concentration;

    double left, right;
    for (unsigned long long int i = 0; i < totalTime; i++) {
        for (unsigned long long int k = 0; k < numSlices; k++) {
            if (k == 0) {
                left = oldCylinder[k];
            } else {
                left = oldCylinder[k - 1];
            }
            if (k == numSlices - 1) {
                right = oldCylinder[k];
            } else {
                right = oldCylinder[k + 1];
            }
            newCylinder[k] = (left + right) / 2.0;
        }
        temp = oldCylinder;
        oldCylinder = newCylinder;
        newCylinder = temp;
    }

    printf("%f\n", oldCylinder[desiredPoint]);

    free(oldCylinder);
    free(newCylinder);
}