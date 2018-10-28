#include <stdio.h>
#include <stdlib.h>

double gpuCalculate(const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int totalTime, const double concentration, const unsigned long long int desiredPointRow, const unsigned long long int desiredPointCol);

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
    if (argc != 7) {
        printf("Usage:\n\t%s <rectangle rows> <rectangle columns> <total time> <point source concentration> <desired point row> <desired point column>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int slicesRow = convertToLLU(argv[1], "Rectangle Rows", 1);
    const unsigned long long int slicesCol = convertToLLU(argv[2], "Rectangle Columns", 1);
    const unsigned long long int totalTime = convertToLLU(argv[3], "Total Time", 0);
    double concentration = convertToDouble(argv[4], "Concentration", 0);
    const unsigned long long int desiredPointRow = convertToLLU(argv[5], "Desired Point Row", 0);
    const unsigned long long int desiredPointCol = convertToLLU(argv[6], "Desired Point Column", 0);

    if (desiredPointRow > slicesRow || desiredPointCol > slicesCol) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }

    double answer = gpuCalculate(slicesRow, slicesCol, totalTime, concentration, desiredPointRow, desiredPointCol);

    printf("%f\n", answer);
}