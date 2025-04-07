#pragma once

#define XDIM 64
#define YDIM 64
#define ZDIM 64
//constexpr for avoiding multiple definitions of constants
constexpr double alpha = 0.05; //buoyancy coefficient (0.05 – 0.1)
constexpr double beta = 0.1; //temperature lift coefficient
constexpr double T_ambient = 0.0; //Ambient temperature
const double dx = 1.0 / (XDIM - 1); // Grid spacing in x-axis
const double dy = 1.0 / (YDIM - 1); // Grid spacing in y-axis
const double dz = 1.0 / (ZDIM - 1); // Grid spacing in z-axis
const double dt = 0.01; //Time increment
