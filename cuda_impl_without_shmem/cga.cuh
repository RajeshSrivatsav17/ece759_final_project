#ifndef CGA_CUH
#define CGA_CUH

__global__ void laplacian_cuda_kernel(float* q, const float* d);
__global__ void initialize_pressure(float* p);
__global__ void compute_residual_kernel(float* r, const float* b, const float* p);
__global__ void update_pressure_and_residual_kernel(float* p, float* r, const float* d, const float* q, float alpha);
__global__ void update_search_direction_kernel(float* d, const float* r, float beta);
__global__ void dot_product_kernel(const float* a, const float* b, float* result, int size);

void solvePressureCG(float* d_p, float* d_b);

#endif