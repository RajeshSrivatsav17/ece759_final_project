#include <iostream>
#include "advect.h"
#include <cmath>

float trilinear_sample(const float (&field) [XDIM][YDIM][ZDIM], float x, float y, float z) {
    int i = static_cast<int>(floor(x));
    int j = static_cast<int>(floor(y));
    int k = static_cast<int>(floor(z));

    float tx = x - i;
    float ty = y - j;
    float tz = z - k;

    auto clamp = [](int v, int minv, int maxv) { return std::max(minv, std::min(v, maxv)); };

    i = clamp(i, 0, XDIM - 2);
    j = clamp(j, 0, YDIM - 2);
    k = clamp(k, 0, ZDIM - 2);

    float c000 = field[i][j][k];
    float c100 = field[i+1][j][k];
    float c010 = field[i][j+1][k];
    float c110 = field[i+1][j+1][k];
    float c001 = field[i][j][k+1];
    float c101 = field[i+1][j][k+1];
    float c011 = field[i][j+1][k+1];
    float c111 = field[i+1][j+1][k+1];

    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

void semi_lagrangian_advection(float (&dst) [XDIM][YDIM][ZDIM], const float (&src) [XDIM][YDIM][ZDIM], const float (&u) [XDIM][YDIM][ZDIM], const float (&v) [XDIM][YDIM][ZDIM], const float(&w) [XDIM][YDIM][ZDIM], float dt) {
    #pragma omp parallel for
    for (int k = 0; k < ZDIM; ++k) {
        for (int j = 0; j < YDIM; ++j) {
            for (int i = 0; i < XDIM; ++i) {

                float x = i - dt * u[i][j][k];
                float y = j - dt * v[i][j][k];
                float z = k - dt * w[i][j][k];

                x = std::max(0.0f, std::min((float)(XDIM - 1.001f), x));
                y = std::max(0.0f, std::min((float)(YDIM - 1.001f), y));
                z = std::max(0.0f, std::min((float)(ZDIM - 1.001f), z));

                dst[i][j][k] = trilinear_sample(src, x, y, z );
            }
        }
    }
}
