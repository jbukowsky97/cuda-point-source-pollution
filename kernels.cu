#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "kernels.cuh"

__global__ void point_source_pollution_kernel(double *old_cylinder, double *new_cylinder, double *density, unsigned int n) {
    
    //the numner of computations each thread needs to do
    unsigned int stride = blockDim.x * gridDim.x;

    //shared cache, idk if this is exactly what we need
    //but we will need some type of shared cache
    __shared__ float cache[256];

    //we want to make sure each thread does the correct 
    //number of computations
    
}