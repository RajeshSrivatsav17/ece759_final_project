#include "Utilities.h"
#include "parameters.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
void buoyantforce(float (&rho) [XDIM][YDIM][ZDIM],float (&T) [XDIM][YDIM][ZDIM],float (&v) [XDIM][YDIM][ZDIM]){
    #pragma omp parallel for
    for(int i = 0; i < XDIM; i++){
        for(int j = 0; j < YDIM; j++){
            for(int k = 0; k < ZDIM; k++){
                double buoy_force = alpha * rho[i][j][k] * beta * (T[i][j][k] - T_ambient);
                v[i][j][k] += buoy_force*dt;
            }
        }
    }
}

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

void buoyantforce(float (&rho) [XDIM][YDIM][ZDIM],float (&T) [XDIM][YDIM][ZDIM],float (&v) [XDIM][YDIM][ZDIM]){
    #pragma omp parallel for
    for(int i = 0; i < XDIM; i++){
        for(int j = 0; j < YDIM; j++){
            for(int k = 0; k < ZDIM; k++){
                double buoy_force = alpha * rho[i][j][k] * beta * (T[i][j][k] - T_ambient);
                v[i][j][k] += buoy_force*dt;
            }
        }
    }
}

void solvePressureCG(
    float (&p)[XDIM][YDIM][ZDIM],
    float (&b)[XDIM][YDIM][ZDIM]
) {
    float* rRaw = new float[XDIM * YDIM * ZDIM]();
    float* dRaw = new float[XDIM * YDIM * ZDIM]();
    float* qRaw = new float[XDIM * YDIM * ZDIM]();

    using array_t = float (&)[XDIM][YDIM][ZDIM];
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t d = reinterpret_cast<array_t>(*dRaw);
    array_t q = reinterpret_cast<array_t>(*qRaw);

    int maxIterations = 100;
    float tolerance = 1e-5f;
    // Initial guess: p = 0
    #pragma omp parallel for
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k)
        p[i][j][k] = 0.0f;

    // r = b - A*p (but p = 0, so r = b)
    #pragma omp parallel for
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k) {
        r[i][j][k] = b[i][j][k];
        d[i][j][k] = r[i][j][k];
    }
    
    float delta_new = 0.0f;
    #pragma omp parallel for reduction(+:delta_new)
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k)
        delta_new += r[i][j][k] * r[i][j][k];

    for (int iter = 0; iter < maxIterations && delta_new > tolerance * tolerance; ++iter) {
        // q = A * d
        #pragma omp parallel for
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            q[i][j][k] = laplacian(d, i, j, k);

        // alpha = delta_new / dot(d, q)
        float dq = 0.0f;
        #pragma omp parallel for reduction(+:dq)
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            dq += d[i][j][k] * q[i][j][k];

        float alpha = delta_new / dq;

        // p = p + alpha * d
        // r = r - alpha * q
        float delta_old = delta_new;
        delta_new = 0.0f;
        #pragma omp parallel for reduction(+:delta_new)
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
        #pragma omp parallel for
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            d[i][j][k] = r[i][j][k] + beta * d[i][j][k];
    }
    delete[] rRaw;
    delete[] dRaw;
    delete[] qRaw;

}

void computeDivergence(
    float (&u)[XDIM][YDIM][ZDIM],
    float (&v)[XDIM][YDIM][ZDIM],
    float (&w)[XDIM][YDIM][ZDIM],
    float (&divergence)[XDIM][YDIM][ZDIM]
)
{
    #pragma omp parallel for
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

void Clear(float (&x)[XDIM][YDIM][ZDIM])
{
#pragma omp parallel for
    for (int i = 0; i < XDIM; i++)
    for (int j = 0; j < YDIM; j++)
    for (int k = 0; k < ZDIM; k++)
        x[i][j][k] = 0.;
}

void InitializeProblem(float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM],float (&z)[XDIM][YDIM][ZDIM]){

    // Start by zeroing out x and b
    Clear(x);
    Clear(y);
    Clear(z);
    /* Set the initial conditions for the density and temperature
    Sets:
    rho = 1.0 → meaning smoke exists in that region
    T = 300.0 → this region is hotter than surrounding cells (used for buoyancy)
    */
    int Z_initial_val = ceil(0.06*ZDIM);
    int Z_limit = ceil(0.12*ZDIM);
    #pragma omp parallel for
    for(int k = Z_initial_val; k < Z_limit; ++k)
        for(int j = YDIM/2-16; j < YDIM/2+16; j++)
        for(int i = XDIM/2-16; i < XDIM/2+16; i++){ //6% to 12% Near the bottom of the domain
            x[i][j][k] = 1.; //Density
            y[i][j][k] = 300.; //Temperature 
            z[i][j][k] = 2.0f;
        }
            
}


void writetoCSV(float (&x)[XDIM][YDIM][ZDIM], const std::string& filename, const std::string& ftype) {
    std::ofstream file(filename);
    file << "x,y,z,density\n";
    for (int k = 0; k < ZDIM; ++k)
    for (int j = 0; j < YDIM; ++j)
    for (int i = 0; i < XDIM; ++i) {
        file << i << "," << j << "," << k << "," << x[i][j][k] << "\n";
    }

    file.close();
}

float laplacian(float (&p)[XDIM][YDIM][ZDIM], int i, int j, int k) {
    float sum = 0.0f;
    if (i > 0)       sum += p[i-1][j][k];
    if (i < XDIM-1)  sum += p[i+1][j][k];
    if (j > 0)       sum += p[i][j-1][k];
    if (j < YDIM-1)  sum += p[i][j+1][k];
    if (k > 0)       sum += p[i][j][k-1];
    if (k < ZDIM-1)  sum += p[i][j][k+1];
    sum -= 6.0f * p[i][j][k];
    return sum / (dx * dx);
}
void applyBoundaryConditions(float (&u)[XDIM][YDIM][ZDIM],
                              float (&v)[XDIM][YDIM][ZDIM],
                              float (&w)[XDIM][YDIM][ZDIM]) {
    #pragma omp parallel for
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k) {
        u[0][j][k] = 0; u[XDIM-1][j][k] = 0; // Applies zero at X boundary velocity
    }
    #pragma omp parallel for
    for (int i = 0; i < XDIM; ++i)
    for (int k = 0; k < ZDIM; ++k) {
        v[i][0][k] = 0; v[i][YDIM-1][k] = 0; // Applies zero at Y boundary velocity
    }
    #pragma omp parallel for
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j) {
        w[i][j][0] = 0; w[i][j][ZDIM-1] = 0; // Applies zero at Z boundary velocity
    }
}