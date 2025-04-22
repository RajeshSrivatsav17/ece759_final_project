#include "velocity_correction.h"

__global__ void velocityCorrection_kernel(float* u, float* v, float* w, float* p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = XDIM * YDIM * ZDIM;

    if (idx >= totalSize) return;

    int i = idx / (YDIM * ZDIM);
    int j = (idx / ZDIM) % YDIM;
    int k = idx % ZDIM;

    if (i > 0 && i < XDIM-1 && j > 0 && j < YDIM-1 && k > 0 && k < ZDIM-1) {
        int idx_center = i * YDIM * ZDIM + j * ZDIM + k;
        int idx_ip = (i+1) * YDIM * ZDIM + j * ZDIM + k;
        int idx_im = (i-1) * YDIM * ZDIM + j * ZDIM + k;
        int idx_jp = i * YDIM * ZDIM + (j+1) * ZDIM + k;
        int idx_jm = i * YDIM * ZDIM + (j-1) * ZDIM + k;
        int idx_kp = i * YDIM * ZDIM + j * ZDIM + (k+1);
        int idx_km = i * YDIM * ZDIM + j * ZDIM + (k-1);

        u[idx_center] -= (p[idx_ip] - p[idx_im]) / (2.0f * dx);
        v[idx_center] -= (p[idx_jp] - p[idx_jm]) / (2.0f * dx);
        w[idx_center] -= (p[idx_kp] - p[idx_km]) / (2.0f * dx);
    }
}
