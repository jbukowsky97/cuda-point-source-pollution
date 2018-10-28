#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void calculateNext(double* oldCylinder, double* newCylinder, const unsigned long long int numSlices) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < numSlices) {
        if (i == 0) {
            newCylinder[i] = (oldCylinder[i] + oldCylinder[i + 1]) / 2.0;
        } else if (i == numSlices - 1) {
            newCylinder[i] = (oldCylinder[i - 1] + oldCylinder[i]) / 2.0;
        } else {
            newCylinder[i] = (oldCylinder[i - 1] + oldCylinder[i + 1]) / 2.0;
        }
    }
}

__global__ void initializeArray(double* cylinder, const unsigned long long int numSlices, const double concentration) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < numSlices) {
        if (i == 0) {
            cylinder[i] = concentration;
        } else {
            cylinder[i] = 0.0;
        }
    }
}

extern "C" double gpuCalculate(const unsigned long long int numSlices, const unsigned long long int totalTime, const double concentration, const unsigned long long int desiredPoint) {
    cudaError_t mallocResult;
    double* oldCylinder;
    double* newCylinder;
    double* temp;

    mallocResult = cudaMalloc((void**) &oldCylinder, numSlices * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMalloc((void**) &newCylinder, numSlices * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    dim3 dimBlock(BLOCK_SIZE);
    unsigned long long int gridSize = ceil(numSlices / (double) BLOCK_SIZE);
    dim3 dimGrid(gridSize);

    initializeArray<<<dimGrid, dimBlock>>>(oldCylinder, numSlices, concentration);

    for (int i = 0; i < totalTime; i++) {
        calculateNext<<<dimGrid, dimBlock>>>(oldCylinder, newCylinder, numSlices);
        temp = oldCylinder;
        oldCylinder = newCylinder;
        newCylinder = temp;
    }

    double answer;
    mallocResult = cudaMemcpy(&answer, &oldCylinder[desiredPoint], sizeof(double), cudaMemcpyDeviceToHost);
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