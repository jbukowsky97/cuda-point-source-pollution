#include <stdio.h>

#define SIDE_SIZE 8

__device__ unsigned long long int cIndex(const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    return slicesRow * row + col;
}

__device__ double cLeft(double* grid, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (col == 0) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row, col - 1)];
}

__device__ double cRight(double* grid, const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int row, const unsigned long long int col) {
    if (col == slicesCol - 1) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row, col + 1)];
}

__device__ double cUp(double* grid, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == 0) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row - 1, col)];
}

__device__ double cDown(double* grid, const unsigned long long int slicesRow, const unsigned long long int row, const unsigned long long int col) {
    if (row == slicesRow - 1) {
        return grid[cIndex(slicesRow, row, col)];
    }
    return grid[cIndex(slicesRow, row + 1, col)];
}

__global__ void calculateNext(double* oldGrid, double* newGrid, const unsigned long long int slicesRow, const unsigned long long int slicesCol) {
    unsigned long long int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row < slicesRow && col < slicesCol) {
        double total = 0.0;
        total += cLeft(oldGrid, slicesRow, row, col);
        total += cRight(oldGrid, slicesRow, slicesCol, row, col);
        total += cUp(oldGrid, slicesRow, row, col);
        total += cDown(oldGrid, slicesRow, row, col);
        newGrid[cIndex(slicesRow, row, col)] = total / 4.0;
    }
}

__global__ void initializeArray(double* cylinder, const unsigned long long int slicesRow, const unsigned long long int slicesCol, const int numImpulses, const unsigned long long int* impulses, const double* concentrations) {
    unsigned long long int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row < slicesRow && col < slicesCol) {
        cylinder[cIndex(slicesRow, row, col)] = 0.0;
        for (int k = 0; k < numImpulses; k++) {
            if (row == impulses[k * 2] && col == impulses[k * 2 + 1]) {
                cylinder[cIndex(slicesRow, impulses[k * 2], impulses[k * 2 + 1])] = concentrations[k];
                break;
            }
        }
    }
}

extern "C" double gpuCalculate(const unsigned long long int slicesRow, const unsigned long long int slicesCol, const unsigned long long int totalTime, const unsigned long long int desiredPointRow, const unsigned long long int desiredPointCol, const int numImpulses, const unsigned long long int* impulses, const double* concentrations) {
    cudaError_t mallocResult;
    double* oldGrid;
    double* newGrid;
    double* temp;

    unsigned long long int* deviceImpulses;
    double* deviceConcentrations;

    mallocResult = cudaMalloc((void**) &oldGrid, slicesRow * slicesCol * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMalloc((void**) &newGrid, slicesRow * slicesCol * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMalloc((void**) &deviceImpulses, 2 * numImpulses * sizeof(unsigned long long int));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMalloc((void**) &deviceConcentrations, numImpulses * sizeof(double));
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMemcpy(deviceImpulses, impulses, 2 * numImpulses * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    mallocResult = cudaMemcpy(deviceConcentrations, concentrations, numImpulses * sizeof(double), cudaMemcpyHostToDevice);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy failed, exiting...\n");
        exit(EXIT_FAILURE);
    }

    dim3 dimBlock(SIDE_SIZE, SIDE_SIZE);
    dim3 dimGrid(ceil(slicesCol / (double) SIDE_SIZE), ceil(slicesRow / (double) SIDE_SIZE));

    initializeArray<<<dimGrid, dimBlock>>>(oldGrid, slicesRow, slicesCol, numImpulses, deviceImpulses, deviceConcentrations);

    for (int i = 0; i < totalTime; i++) {
        calculateNext<<<dimGrid, dimBlock>>>(oldGrid, newGrid, slicesRow, slicesCol);
        temp = oldGrid;
        oldGrid = newGrid;
        newGrid = temp;
    }

    double answer;
    mallocResult = cudaMemcpy(&answer, &oldGrid[slicesRow * desiredPointRow + desiredPointCol], sizeof(double), cudaMemcpyDeviceToHost);
    if (mallocResult != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy failed, exiting...\n");
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