#include "parameters.h"
void semi_lagrangian_advection(float (&dst) [XDIM][YDIM][ZDIM], const float (&src) [XDIM][YDIM][ZDIM], const float (&u) [XDIM][YDIM][ZDIM], const float (&v) [XDIM][YDIM][ZDIM], const float(&w) [XDIM][YDIM][ZDIM], float dt);
