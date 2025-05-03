#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#include <string>
#include "parameters.h"

// Function to write 3D data to a CSV file
void writetoCSV(const float* x, const std::string& filename, const std::string& ftype);

#endif // CPU_FUNCTIONS_H