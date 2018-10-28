#include <stdio.h>
#include <stdlib.h>

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

double cLeft(double* container, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (col == 0) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row, col - 1)];
}

double cRight(double* container, const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int row, const unsigned long long int col) {
    if (col == slicesCol - 1) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row, col + 1)];
}

double cUp(double* container, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == 0) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row - 1, col)];
}

double cDown(double* container, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == slicesRow - 1) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row + 1, col)];
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

    double* oldCylinder = (double*) calloc(slicesRow * slicesCol, sizeof(double));
    double* newCylinder = (double*) malloc(slicesRow * slicesCol * sizeof(double));
    double* temp;

    if (oldCylinder == NULL || newCylinder == NULL) {
        printf("Error while allocating memory\n");
        exit(EXIT_FAILURE);
    }

    oldCylinder[0] = concentration;

    double total;
    for (unsigned long long int i = 0; i < totalTime; i++) {
        for (unsigned long long int row = 0; row < slicesRow; row++) {
            for (unsigned long long int col = 0; col < slicesCol; col++) {
                total = 0.0;
                total += cLeft(oldCylinder, slicesRow, row, col);
                total += cRight(oldCylinder, slicesRow, slicesCol, row, col);
                total += cUp(oldCylinder, slicesRow, row, col);
                total += cDown(oldCylinder, slicesRow, row, col);
                newCylinder[cIndex(slicesRow, row, col)] = total / 4.0;
            }
        }
        temp = oldCylinder;
        oldCylinder = newCylinder;
        newCylinder = temp;
    }

    printf("%f\n", oldCylinder[cIndex(slicesRow, desiredPointRow, desiredPointCol)]);

    free(oldCylinder);
    free(newCylinder);
}