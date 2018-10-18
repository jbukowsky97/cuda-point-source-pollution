#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage:\n\t%s <cylinder size> <slice width> <time unit> <total time> <point source concentration>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int cylinderSize = strtoull(argv[1], NULL, 0);
    if (cylinderSize < 0) {
        printf("Cylinder size must be > 0\n");
    }

    const unsigned long long int sliceWidth = strtoull(argv[2], NULL, 0);
    if (sliceWidth < 0) {
        printf("Slice width must be > 0\n");
    }

    const unsigned long long int timeUnit = strtoull(argv[3], NULL, 0);
    if (timeUnit < 0) {
        printf("Time unit must be > 0\n");
    }

    const unsigned long long int totalTime = strtoull(argv[4], NULL, 0);
    if (totalTime < 0) {
        printf("Total time must be > 0\n");
    }

    const unsigned long long int concentration = strtoull(argv[5], NULL, 0);
    if (concentration < 0) {
        printf("Concentration must be > 0\n");
    }
}