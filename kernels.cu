#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "kernels.cuh"

__global__ void point_source_pollution_kernel(double *old_cylinder, double *new_cylinder, double *density, unsigned int n) {
    //write our parallelized algorithm here amiright?
    int x = 0;
}