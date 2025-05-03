#include <cmath>
#include "advect.h"

__device__ float trilinear_sample_shared(const float *shared_field, int x, int y, int z, float tx, float ty, float tz, int shared_dim) {
    // Clamp indices to shared memory boundaries
    x = max(0, min(x, shared_dim - 2));
    y = max(0, min(y, shared_dim - 2));
    z = max(0, min(z, shared_dim - 2));

    // Compute indices for trilinear interpolation
    int idx000 = z + y * shared_dim + x * shared_dim * shared_dim;
    int idx100 = z + y * shared_dim + (x + 1) * shared_dim * shared_dim;
    int idx010 = z + (y + 1) * shared_dim + x * shared_dim * shared_dim;
    int idx110 = z + (y + 1) * shared_dim + (x + 1) * shared_dim * shared_dim;
    int idx001 = (z + 1) + y * shared_dim + x * shared_dim * shared_dim;
    int idx101 = (z + 1) + y * shared_dim + (x + 1) * shared_dim * shared_dim;
    int idx011 = (z + 1) + (y + 1) * shared_dim + x * shared_dim * shared_dim;
    int idx111 = (z + 1) + (y + 1) * shared_dim + (x + 1) * shared_dim * shared_dim;

    // Perform trilinear interpolation
    float c00 = shared_field[idx000] * (1 - tx) + shared_field[idx100] * tx;
    float c10 = shared_field[idx010] * (1 - tx) + shared_field[idx110] * tx;
    float c01 = shared_field[idx001] * (1 - tx) + shared_field[idx101] * tx;
    float c11 = shared_field[idx011] * (1 - tx) + shared_field[idx111] * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

__global__ void semi_lagrangian_advection_kernel(float *output, const float *input, const float *u, const float *v, const float *w, float dt) {
    extern __shared__ float shared_mem[]; // Shared memory allocation
    float *shared_input = shared_mem;    // Shared memory for input field

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int local_z = threadIdx.z;

    int shared_dim = blockDim.x; // Assuming cubic blocks

    // Load data into shared memory
    if (x < XDIM && y < YDIM && z < ZDIM) {
        shared_input[local_z + local_y * shared_dim + local_x * shared_dim * shared_dim] =
            input[z + y * ZDIM + x * YDIM * ZDIM];
    }
    __syncthreads();

    // Compute backtraced position
    if (x < XDIM && y < YDIM && z < ZDIM) {
        float x_back = x - dt * u[z + y * ZDIM + x * YDIM * ZDIM];
        float y_back = y - dt * v[z + y * ZDIM + x * YDIM * ZDIM];
        float z_back = z - dt * w[z + y * ZDIM + x * YDIM * ZDIM];

        // Clamp to grid boundaries
        x_back = max(0.0f, min((float)(XDIM - 1), x_back));
        y_back = max(0.0f, min((float)(YDIM - 1), y_back));
        z_back = max(0.0f, min((float)(ZDIM - 1), z_back));

        // Compute integer and fractional parts of the backtraced position
        int x0 = floor(x_back);
        int y0 = floor(y_back);
        int z0 = floor(z_back);

        float tx = x_back - x0;
        float ty = y_back - y0;
        float tz = z_back - z0;

        // Perform trilinear interpolation using shared memory
        float interpolated_value = trilinear_sample_shared(shared_input, x0, y0, z0, tx, ty, tz, shared_dim);

        // Write the result to the output field
        output[z + y * ZDIM + x * YDIM * ZDIM] = interpolated_value;
    }
}