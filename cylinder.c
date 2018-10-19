#include <stdio.h>
#include <stdlib.h>

unsigned long long int convertToNum(char* str) {
    const unsigned long long int temp = strtoull(str, NULL, 0);
    if (temp < 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        printf("Usage:\n\t%s <cylinder size> <slice width> <time unit> <total time> <point source concentration> <desired point>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int cylinderSize = convertToNum(argv[1]);
    const unsigned long long int sliceWidth = convertToNum(argv[2]);
    const unsigned long long int timeUnit = convertToNum(argv[3]);
    const unsigned long long int totalTime = convertToNum(argv[4]);
    const unsigned long long int concentration = convertToNum(argv[5]);
    const unsigned long long int desiredPoint = convertToNum(argv[6]);

    if (timeUnit > totalTime) {
        printf("Time unit must be less than total time\n");
        exit(EXIT_FAILURE);
    }

    if (desiredPoint > cylinderSize) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }
}