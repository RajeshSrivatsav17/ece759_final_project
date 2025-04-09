#include "divergence.h"

void computeDivergence(
    float (&u)[XDIM][YDIM][ZDIM],
    float (&v)[XDIM][YDIM][ZDIM],
    float (&w)[XDIM][YDIM][ZDIM],
    float (&divergence)[XDIM][YDIM][ZDIM]
)
{
    for (int i = 1; i < XDIM - 1; i++) {
        for (int j = 1; j < YDIM - 1; j++) {
            for (int k = 1; k < ZDIM - 1; k++) {
                float du_dx = (u[i + 1][j][k] - u[i - 1][j][k]) / (2.0f * dx);
                float dv_dy = (v[i][j + 1][k] - v[i][j - 1][k]) / (2.0f * dx);
                float dw_dz = (w[i][j][k + 1] - w[i][j][k - 1]) / (2.0f * dx);

                divergence[i][j][k] = du_dx + dv_dy + dw_dz;
            }
        }
    }
}
