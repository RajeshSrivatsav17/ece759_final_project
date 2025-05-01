#include "divergence.h"

__global__ void computeDivergence_kernel(const float *u, const float *v, const float *w, float *divergence)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalSize = XDIM * YDIM * ZDIM;

    if (idx < totalSize) {
        // For Z-major layout:
        int x = idx / (YDIM * ZDIM);
        int y = (idx / ZDIM) % YDIM;
        int z = idx % ZDIM;

        if (x >= 1 && x < XDIM - 1 &&
            y >= 1 && y < YDIM - 1 &&
            z >= 1 && z < ZDIM - 1) {

            // Z-major index: z + y * ZDIM + x * YDIM * ZDIM
            auto index = [](int x, int y, int z) {
                return z + y * ZDIM + x * YDIM * ZDIM;
            };

            float du_dx = (u[index(x + 1, y, z)] - u[index(x - 1, y, z)]) / (2.0f * dx);
            float dv_dy = (v[index(x, y + 1, z)] - v[index(x, y - 1, z)]) / (2.0f * dx);
            float dw_dz = (w[index(x, y, z + 1)] - w[index(x, y, z - 1)]) / (2.0f * dx);

            divergence[idx] = du_dx + dv_dy + dw_dz;
        }
    }
}

