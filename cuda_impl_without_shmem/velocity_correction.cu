#include "velocity_correction.h"

#define IDX(i, j, k) ((k) + (j) * ZDIM + (i) * YDIM * ZDIM)

__global__ void velocityCorrection_kernel(float* u, float* v, float* w, float* p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = XDIM * YDIM * ZDIM;

    if (idx >= totalSize) return;

    int i = idx / (YDIM * ZDIM);
    int j = (idx / ZDIM) % YDIM;
    int k = idx % ZDIM;

    // Ensure we are not on boundary cells
    if (i > 0 && i < XDIM - 1 &&
        j > 0 && j < YDIM - 1 &&
        k > 0 && k < ZDIM - 1) {

        int idx_center = IDX(i, j, k);
        int idx_ip = IDX(i + 1, j, k);
        int idx_im = IDX(i - 1, j, k);
        int idx_jp = IDX(i, j + 1, k);
        int idx_jm = IDX(i, j - 1, k);
        int idx_kp = IDX(i, j, k + 1);
        int idx_km = IDX(i, j, k - 1);

        // Apply velocity correction using pressure gradient
        u[idx_center] -= (p[idx_ip] - p[idx_im]) / (2.0f * dx);
        v[idx_center] -= (p[idx_jp] - p[idx_jm]) / (2.0f * dx);
        w[idx_center] -= (p[idx_kp] - p[idx_km]) / (2.0f * dx);
    }
}

