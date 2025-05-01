#ifndef CGA_CUH
#define CGA_CUH

__global__ void cg_pressure_solver_flat(
    float* p, const float* b, float dx, int N, int maxIters, float tolerance);

    void solvePressureCG(float* d_p, float* d_b) ;

#endif