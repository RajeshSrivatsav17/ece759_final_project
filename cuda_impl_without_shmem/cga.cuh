#ifndef CGA_CUH
#define CGA_CUH

void solvePressureCG_CUDA(
    float* p, float* b, int xdim, int ydim, int zdim, int maxIterations, float tolerance);

#endif