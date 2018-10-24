#include <stdio.h>
#include <stdlib.h>

double gpuCalculate(const unsigned long long int numSlices, const unsigned long long int totalTime, const double concentration, const unsigned long long int desiredPoint);

unsigned long long int convertToNum(char* str) {
    const unsigned long long int temp = strtoull(str, NULL, 0);
    if (temp <= 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    return temp;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage:\n\t%s <cylinder size> <slice width> <total time> <point source concentration> <desired point>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int cylinderSize = convertToNum(argv[1]);
    const unsigned long long int sliceWidth = convertToNum(argv[2]);
    const unsigned long long int totalTime = convertToNum(argv[3]);
    double concentration = atof(argv[4]);
    if (concentration < 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    const unsigned long long int desiredPoint = convertToNum(argv[5]);

    if (desiredPoint > cylinderSize) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }

    if (cylinderSize % sliceWidth != 0) {
        printf("Slice width must evenly divide cylinder size\n");
        exit(EXIT_FAILURE);
    }

    const unsigned long long int numSlices = cylinderSize / sliceWidth;

    double answer = gpuCalculate(numSlices, totalTime, concentration, desiredPoint);
    printf("%f\n", answer);
}