#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "parameters.h"

void Clear(float *x);
void InitializeProblem(float (&x)[XDIM][YDIM][ZDIM], float (&b)[XDIM][YDIM][ZDIM], float (&c)[XDIM][YDIM][ZDIM]);
void writetoCSV(float (&x)[XDIM][YDIM][ZDIM], const std::string& filename, const std::string& ftype);
float laplacian(float (&p)[XDIM][YDIM][ZDIM], int i, int j, int k);