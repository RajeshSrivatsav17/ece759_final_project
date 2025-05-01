#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cg_pressure_solver_flat(
    float* p, const float* b, float dx, int N, int maxIters, float tolerance)
{
    extern __shared__ float shared[];

    float* r = shared;                   // [0, N)
    float* d = &shared[N];               // [N, 2N)
    float* q = &shared[2 * N];           // [2N, 3N)

    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float delta_new = 0.0f;

    if (idx < N) {
        p[idx] = 0.0f;
        r[idx] = b[idx];
        d[idx] = r[idx];
    }
    __syncthreads();

    // Initial delta_new = dot(r, r)
    float local_dot = 0.0f;
    if (idx < N) local_dot = r[idx] * r[idx];

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
    
    static __shared__ float global_dot;
    if (threadIdx.x % warpSize == 0)
        atomicAdd(&global_dot, local_dot);
    __syncthreads();

    if (threadIdx.x == 0) delta_new = global_dot;
    __syncthreads();
    delta_new = global_dot;

    float delta_old;

    for (int iter = 0; iter < maxIters && delta_new > tolerance * tolerance; ++iter) {
        if (threadIdx.x == 0) global_dot = 0.0f;
        __syncthreads();

        // q = Laplacian(d)
        if (idx < N) {
            int i = idx / (YDIM * ZDIM);
            int j = (idx / ZDIM) % YDIM;
            int k = idx % ZDIM;

            float center = d[idx];
            float sum = 0.0f;

            if (i > 0)        sum += d[IDX(i - 1, j, k)];
            if (i < XDIM - 1) sum += d[IDX(i + 1, j, k)];
            if (j > 0)        sum += d[IDX(i, j - 1, k)];
            if (j < YDIM - 1) sum += d[IDX(i, j + 1, k)];
            if (k > 0)        sum += d[IDX(i, j, k - 1)];
            if (k < ZDIM - 1) sum += d[IDX(i, j, k + 1)];

            q[idx] = (sum - 6.0f * center) / (dx * dx);
        }
        __syncthreads();

        // dq = dot(d, q)
        float dq_local = 0.0f;
        if (idx < N) dq_local = d[idx] * q[idx];

        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            dq_local += __shfl_down_sync(0xffffffff, dq_local, offset);

        if (threadIdx.x % warpSize == 0)
            atomicAdd(&global_dot, dq_local);
        __syncthreads();

        float dq = global_dot;
        float alpha = delta_new / dq;

        // Update p and r
        if (idx < N) {
            p[idx] += alpha * d[idx];
            r[idx] -= alpha * q[idx];
        }
        __syncthreads();

        if (threadIdx.x == 0) global_dot = 0.0f;
        __syncthreads();

        // delta_old = delta_new
        delta_old = delta_new;

        // delta_new = dot(r, r)
        float dot_local = 0.0f;
        if (idx < N) dot_local = r[idx] * r[idx];

        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            dot_local += __shfl_down_sync(0xffffffff, dot_local, offset);

        if (threadIdx.x % warpSize == 0)
            atomicAdd(&global_dot, dot_local);
        __syncthreads();

        delta_new = global_dot;

        // beta = delta_new / delta_old
        float beta = delta_new / delta_old;

        // Update d = r + beta * d
        if (idx < N) {
            d[idx] = r[idx] + beta * d[idx];
        }
        __syncthreads();
    }
}

void solvePressureCG(float* d_p, float* d_b) {
    int N = XDIM * YDIM * ZDIM;
    int maxIters = 100;
    float tolerance = 1e-5f;

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    size_t sharedMemSize = 3 * N * sizeof(float);  // r, d, q

    void* args[] = { &d_p, &d_b, &dx, &N, &maxIters, &tolerance };

    cudaLaunchCooperativeKernel(
        (void*)cg_pressure_solver_flat, gridSize, blockSize, args, sharedMemSize);
}


