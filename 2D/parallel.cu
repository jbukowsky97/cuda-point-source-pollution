#include <stdio.h>

#define SIDE_SIZE 8

__device__ unsigned long long int cIndex(const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    return slicesRow * row + col;
}

__device__ double cLeft(double* container, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (col == 0) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row, col - 1)];
}

__device__ double cRight(double* container, const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int row, const unsigned long long int col) {
    if (col == slicesCol - 1) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row, col + 1)];
}

__device__ double cUp(double* container, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == 0) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row - 1, col)];
}

__device__ double cDown(double* container, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == slicesRow - 1) {
        return container[cIndex(slicesRow, row, col)];
    }
    return container[cIndex(slicesRow, row + 1, col)];
}

__global__ void calculateNext(double* oldCylinder, double* newCylinder, const unsigned long long int slicesRow, const unsigned long long int slicesCol) {
    unsigned long long int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row < slicesRow && col < slicesCol) {
        double total = 0.0;
        total += cLeft(oldCylinder, slicesRow, row, col);
        total += cRight(oldCylinder, slicesRow, slicesCol, row, col);
        total += cUp(oldCylinder, slicesRow, row, col);
        total += cDown(oldCylinder, slicesRow, row, col);
        newCylinder[cIndex(slicesRow, row, col)] = total / 4.0;
    }
}

__global__ void initializeArray(double* cylinder, const unsigned long long int slicesRow, const unsigned long long int slicesCol, const double concentration) {
    unsigned long long int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row < slicesRow && col < slicesCol) {
        if (row == 0 && col == 0) {
            cylinder[cIndex(slicesRow, row, col)] = concentration;
        } else {
            cylinder[cIndex(slicesRow, row, col)] = 0.0;
        }
    }
}

extern "C" double gpuCalculate(const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int totalTime, const double concentration, const unsigned long long int desiredPointRow, const unsigned long long int desiredPointCol) {
    cudaError_t mallocResult;
    double* oldCylinder;
    double* newCylinder;
    double* temp;

    mallocResult = cudaMalloc((void**) &oldCylinder, slicesRow * slicesCol * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMalloc((void**) &newCylinder, slicesRow * slicesCol * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    dim3 dimBlock(SIDE_SIZE, SIDE_SIZE);
    dim3 dimGrid(ceil(slicesCol / (double) SIDE_SIZE), ceil(slicesRow / (double) SIDE_SIZE));

    initializeArray<<<dimGrid, dimBlock>>>(oldCylinder, slicesRow, slicesCol, concentration);

    for (int i = 0; i < totalTime; i++) {
        calculateNext<<<dimGrid, dimBlock>>>(oldCylinder, newCylinder, slicesRow, slicesCol);
        temp = oldCylinder;
        oldCylinder = newCylinder;
        newCylinder = temp;
    }

    double answer;
    mallocResult = cudaMemcpy(&answer, &oldCylinder[slicesRow * desiredPointRow + desiredPointCol], sizeof(double), cudaMemcpyDeviceToHost);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaFree(oldCylinder);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA free failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaFree(newCylinder);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA free failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    return answer;
}
