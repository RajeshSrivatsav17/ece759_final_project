#include "divergence.h"
#include "parameters.h"

#define IDX(i, j, k) ((i) * YDIM * ZDIM + (j) * ZDIM + (k))

__global__ void computeDivergence_kernel(const float *u, const float *v, const float *w, float *divergence)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalSize = XDIM * YDIM * ZDIM;

    if (idx < totalSize) {
        int i = idx / (YDIM * ZDIM);          // x-index
        int j = (idx / ZDIM) % YDIM;          // y-index
        int k = idx % ZDIM;                   // z-index

        if (i >= 1 && i < XDIM - 1 && j >= 1 && j < YDIM - 1 && k >= 1 && k < ZDIM - 1) {
            float du_dx = (u[IDX(i + 1, j, k)] - u[IDX(i - 1, j, k)]) / (2.0f * dx);
            float dv_dy = (v[IDX(i, j + 1, k)] - v[IDX(i, j - 1, k)]) / (2.0f * dx);
            float dw_dz = (w[IDX(i, j, k + 1)] - w[IDX(i, j, k - 1)]) / (2.0f * dx);
            divergence[idx] = du_dx + dv_dy + dw_dz;
        }
    }
}
