#include "divergence.h"

__global__ void computeDivergence_kernel(const float *u, const float *v, const float *w, float *divergence)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalSize = XDIM * YDIM * ZDIM;
    if(idx < totalSize){
        // For 3D blockDim significant change needed 
        int x = idx % XDIM;  // x iterator
        int y = (idx / XDIM) % YDIM; // y iterator logic , skips x values and gets y reminder
        int z = idx / (XDIM * YDIM); // z iterator logic , skips x y valyes and  gets z
        if( x >= 1 && x < XDIM - 1 && y >= 1 && y < YDIM - 1 && z >= 1 && z < ZDIM - 1){
            
            int u1 = ( x - 1 ) + y * XDIM + z * XDIM * YDIM;
            int u2 = ( x + 1 ) + y * XDIM + z * XDIM * YDIM;
            float du_dx = ( u[u2] - u[u1] ) / ( 2.0f * dx);

            int v1 = x + ( y - 1 ) * XDIM + z * XDIM * YDIM;
            int v2 = x + ( y + 1 ) * XDIM + z * XDIM * YDIM;
            float dv_dy = (v[v2] - v[v1]) / (2.0f * dx);

            int w1 = x  + y * XDIM + ( z - 1 ) * XDIM * YDIM;
            int w2 = x  + y * XDIM + ( z + 1 ) * XDIM * YDIM;
            float dw_dz = (w[w2] - w[w1]) / (2.0f * dx);          
  
            divergence[idx] = du_dx + dv_dy + dw_dz; 

        }          
    }
    return;
}
