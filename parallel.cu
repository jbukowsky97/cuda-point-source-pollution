#include <stdio.h>

#define BLOCK_SIZE 128

__global__ void calculateNext(double* oldCylinder, double* newCylinder, const unsigned long long int numSlices, const unsigned long long int totalTime) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double* temp;
    for (unsigned long long int t = 0; t < totalTime; t++) {
        if (i < numSlices) {
            double left;
            double right;
            if (i == 0) {
                left = oldCylinder[i];
            } else {
                left = oldCylinder[i - 1];
            }
            if (i == numSlices - 1) {
                right = oldCylinder[i];
            } else {
                right = oldCylinder[i + 1];
            }
            newCylinder[i] = (left + right) / 2.0;
        }
        temp = oldCylinder;
        oldCylinder = newCylinder;
        newCylinder = temp;
        __syncthreads();
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

    calculateNext<<<dimGrid, dimBlock>>>(oldCylinder, newCylinder, numSlices, totalTime);

    if (totalTime % 2 != 0) {
        double* temp = oldCylinder;
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