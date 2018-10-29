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

__global__ void initializeArray(double* cylinder, const unsigned long long int numSlices, const int numImpulses, const unsigned long long int* impulses, const double* concentrations) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < numSlices) {
        cylinder[i] = 0.0;
        for (int k = 0; k < numImpulses; k++) {
            if (i == impulses[k]) {
                cylinder[i] = concentrations[k];
                break;
            }
        }
    }
}

extern "C" double gpuCalculate(const unsigned long long int numSlices, const unsigned long long int totalTime, const unsigned long long int desiredPoint, const int numImpulses, const unsigned long long int* impulses, const double* concentrations) {
    cudaError_t mallocResult;
    double* oldCylinder;
    double* newCylinder;
    double* temp;

    unsigned long long int* deviceImpulses;
    double* deviceConcentrations;

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

    mallocResult = cudaMalloc((void**) &deviceImpulses, numImpulses * sizeof(unsigned long long int));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMalloc((void**) &deviceConcentrations, numImpulses * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMemcpy(deviceImpulses, impulses, numImpulses * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMemcpy(deviceConcentrations, concentrations, numImpulses * sizeof(double), cudaMemcpyHostToDevice);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    dim3 dimBlock(BLOCK_SIZE);
    unsigned long long int gridSize = ceil(numSlices / (double) BLOCK_SIZE);
    dim3 dimGrid(gridSize);

    initializeArray<<<dimGrid, dimBlock>>>(oldCylinder, numSlices, numImpulses, deviceImpulses, deviceConcentrations);

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

    mallocResult = cudaFree(deviceImpulses);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA free failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaFree(deviceConcentrations);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA free failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    return answer;
}