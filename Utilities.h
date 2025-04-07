#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "parameters.h"

void Clear(float (&x)[XDIM][YDIM][ZDIM]);
void InitializeProblem(float (&x)[XDIM][YDIM][ZDIM], float (&b)[XDIM][YDIM][ZDIM]);
void writetoCSV(float (&x)[XDIM][YDIM][ZDIM], const std::string& filename);

