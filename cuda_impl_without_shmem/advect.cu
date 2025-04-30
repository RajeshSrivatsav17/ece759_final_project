#include <cmath>
#include "advect.h"

__device__ float trilinear_sample(const float *field, float x, float y, float z) {
    int i = static_cast<int>(floorf(x));
    int j = static_cast<int>(floorf(y));
    int k = static_cast<int>(floorf(z));

    float tx = x - i;
    float ty = y - j;
    float tz = z - k;

    i = max(0, min(i, XDIM - 2));
    j = max(0, min(j, YDIM - 2));
    k = max(0, min(k, ZDIM - 2));

    float c000 = field[i * YDIM * ZDIM + j * ZDIM + k];
    float c100 = field[(i+1) * YDIM * ZDIM + j * ZDIM + k];
    float c010 = field[i * YDIM * ZDIM + (j+1) * ZDIM + k];
    float c110 = field[(i+1) * YDIM * ZDIM + (j+1) * ZDIM + k];
    float c001 = field[i * YDIM * ZDIM + j * ZDIM + (k+1)];
    float c101 = field[(i+1) * YDIM * ZDIM + j * ZDIM + (k+1)];
    float c011 = field[i * YDIM * ZDIM + (j+1) * ZDIM + (k+1)];
    float c111 = field[(i+1) * YDIM * ZDIM + (j+1) * ZDIM + (k+1)];
    
    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

__global__ void semi_lagrangian_advection_kernel(float *dst, const float *src, const float *u, const float *v, const float *w, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalSize = XDIM * YDIM * ZDIM;

    if (idx < totalSize) {
        int x = idx / (YDIM * ZDIM);              // x index
        int y = (idx / ZDIM) % YDIM;              // y index
        int z = idx % ZDIM;                       // z index

        float xf = x - dt * u[idx];
        float yf = y - dt * v[idx];
        float zf = z - dt * w[idx];

        xf = fmaxf(0.0f, fminf((float)(XDIM - 1.001f), xf));
        yf = fmaxf(0.0f, fminf((float)(YDIM - 1.001f), yf));
        zf = fmaxf(0.0f, fminf((float)(ZDIM - 1.001f), zf));

        dst[idx] = trilinear_sample(src, xf, yf, zf);
    }
}