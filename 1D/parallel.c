#include <stdio.h>
#include <stdlib.h>

double gpuCalculate(const unsigned long long int numSlices, const unsigned long long int totalTime, const double concentration, const unsigned long long int desiredPoint);

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
    if (argc != 5) {
        printf("Usage:\n\t%s <cylinder slices> <total time> <point source concentration> <desired point>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int numSlices = convertToLLU(argv[1], "Cylinder Slices", 1);
    const unsigned long long int totalTime = convertToLLU(argv[2], "Total Time", 0);
    double concentration = convertToDouble(argv[3], "Concentration", 0);
    const unsigned long long int desiredPoint = convertToLLU(argv[4], "Desired Point", 0);

    if (desiredPoint > numSlices) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }

    double answer = gpuCalculate(numSlices, totalTime, concentration, desiredPoint);
    printf("%f\n", answer);
}