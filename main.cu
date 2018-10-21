#include <stdio.h>
#include <stdlib.h>

unsigned long long int convertToNum(char* str) {
    const unsigned long long int temp = strtoull(str, NULL, 0);
    if (temp <= 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    return temp;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage:\n\t%s <cylinder size> <slice width> <total time> <point source concentration> <desired point>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned long long int cylinderSize = convertToNum(argv[1]);
    const unsigned long long int sliceWidth = convertToNum(argv[2]);
    const unsigned long long int totalTime = convertToNum(argv[3]);
    double concentration = atof(argv[4]);

    if (concentration < 0) {
        printf("Inputs must be greater than 0\n");
        exit(EXIT_FAILURE);
    }

    const unsigned long long int desiredPoint = convertToNum(argv[5]);
    
    if (desiredPoint > cylinderSize) {
        printf("Desired point must be less than cylinder size\n");
        exit(EXIT_FAILURE);
    }

    if (cylinderSize % sliceWidth != 0) {
        printf("Slice width must evenly divide cylinder size\n");
        exit(EXIT_FAILURE);
    }

    const unsigned long long int numSlices = cylinderSize / sliceWidth;


    unsigned int n = 1000 * 256 * 256;
    double *host_density;
    double *device_density;
    double *host_old_cylinder, *host_new_cylinder;
    double *device_old_cylinder, *device_new_cylinder;

    //allocate memory
    host_density = (double*)malloc(sizeof(double));
    host_old_cylinder = (double*) calloc(numSlices, sizeof(double));
    host_new_cylinder = (double*) malloc(numSlices * sizeof(double));

    //allocate memory for GPU
    cudaMalloc((void**)&device_density, sizeof(double);
    cudaMalloc((void**)&device_old_cylinder, sizeof(double);
    cudaMalloc((void**)&device_new_cylinder, sizeof(double);
    cudaMemset(device_density, 0.0, sizeof(double));

    double* temp;

    if (host_old_cylinder == NULL || host_new_cylinder == NULL) {
        printf("Error while allocating memory\n");
        exit(EXIT_FAILURE);
    }

    host_old_cylinder[0] = concentration;

    //timing variables
    float device_elapsed_time = 0.0;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    //copy data to GPU
    cudaMemcpy(device_old_cylinder, host_old_cylinder, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_new_cylinder, host_new_cylinder, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(gpu_start, 0);

    //call kernel
    dim3 gridSize = 256;
    dim3 blockSize = 256;
    point_source_pollution_kernel<<<gridSize, blockSize>>>(host_old_cylinder, host_new_cylinder, device_density, n);

    //copy data back to host
    cudaEventRecord(gpu_stop, 0);
    cudaEventSyncronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaMemcpy(host_density, device_density, sizeof(double), cudaMemcpyDeviceToHost);

    //report results
    std::cout<<"density computed on GPU is: "<<*host_density<<" and took "<<device_elapsed_time<<std::endl;


    double left, right;
    for (unsigned long long int i = 0; i < totalTime; i++) {
        for (unsigned long long int k = 0; k < numSlices; k++) {
            if (k == 0) {
                left = host_old_cylinder[k];
            } else {
                left = host_old_cylinder[k - 1];
            }
            if (k == numSlices - 1) {
                right = host_old_cylinder[k];
            } else {
                right = host_old_cylinder[k + 1];
            }
            host_new_cylinder[k] = (left + right) / 2.0;
        }
        temp = host_old_cylinder;
        host_old_cylinder = host_new_cylinder;
        host_new_cylinder = temp;
    }

    printf("%f\n", host_old_cylinder[desiredPoint]);

    double fudge = 0.0;
    for (int i = 0; i < numSlices; i++) {
        fudge += host_old_cylinder[i];
    }
    printf("total:\t%f\n", fudge);

    free(host_old_cylinder);
    free(host_new_cylinder);
}