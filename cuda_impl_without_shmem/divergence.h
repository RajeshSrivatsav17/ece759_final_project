#include <iostream>
#include "parameters.h"

void computeDivergence(
    float (&u)[XDIM][YDIM][ZDIM],
    float (&v)[XDIM][YDIM][ZDIM],
    float (&w)[XDIM][YDIM][ZDIM],
    float (&divergence)[XDIM][YDIM][ZDIM]
);
