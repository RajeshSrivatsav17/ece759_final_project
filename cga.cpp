#include "cga.h"

void solvePressureCG(
    float (&p)[XDIM][YDIM][ZDIM],
    float (&b)[XDIM][YDIM][ZDIM]
) {
    float r[XDIM][YDIM][ZDIM] = {};
    float d[XDIM][YDIM][ZDIM] = {};
    float q[XDIM][YDIM][ZDIM] = {};

    int maxIterations = 100;
    float tolerance = 1e-5f;
    // Initial guess: p = 0
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k)
        p[i][j][k] = 0.0f;

    // r = b - A*p (but p = 0, so r = b)
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k) {
        r[i][j][k] = b[i][j][k];
        d[i][j][k] = r[i][j][k];
    }
    
    float delta_new = 0.0f;
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k)
        delta_new += r[i][j][k] * r[i][j][k];

    for (int iter = 0; iter < maxIterations && delta_new > tolerance * tolerance; ++iter) {
        // q = A * d
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            q[i][j][k] = laplacian(d, i, j, k);

        // alpha = delta_new / dot(d, q)
        float dq = 0.0f;
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            dq += d[i][j][k] * q[i][j][k];

        float alpha = delta_new / dq;

        // p = p + alpha * d
        // r = r - alpha * q
        float delta_old = delta_new;
        delta_new = 0.0f;
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k) {
            p[i][j][k] += alpha * d[i][j][k];
            r[i][j][k] -= alpha * q[i][j][k];
            delta_new += r[i][j][k] * r[i][j][k];
        }

        // beta = delta_new / delta_old
        float beta = delta_new / delta_old;

        // d = r + beta * d
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            d[i][j][k] = r[i][j][k] + beta * d[i][j][k];
    }
}
