#include <stdio.h>
#include <stdlib.h>

double gpuCalculate(const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int totalTime, const double concentration, const unsigned long long int desiredPointRow, const unsigned long long int desiredPointCol);

unsigned long long int convertToNum(char* str) {
    const unsigned long long int temp = strtoull(str, NULL, 0);
    if (temp < 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    return temp;
}

unsigned long long int cIndex(const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    return slicesRow * row + col;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        printf("Usage:\n\t%s <rectangle rows> <rectangle columns> <total time> <point source concentration> <desired point row> <desired point column>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int slicesRow = convertToNum(argv[1]);
    const unsigned long long int slicesCol = convertToNum(argv[2]);
    const unsigned long long int totalTime = convertToNum(argv[3]);
    double concentration = atof(argv[4]);
    if (concentration <= 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    const unsigned long long int desiredPointRow = convertToNum(argv[5]);
    const unsigned long long int desiredPointCol = convertToNum(argv[6]);

    if (desiredPointRow > slicesRow || desiredPointCol > slicesCol) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }

    double answer = gpuCalculate(slicesRow, slicesCol, totalTime, concentration, desiredPointRow, desiredPointCol);

    printf("%f\n", answer);
}