#include <fstream>
#include <string>
#include "parameters.h"

void writetoCSV(const float* x, const std::string& filename, const std::string& ftype) {
    std::ofstream file(filename);
    file << "x,y,z," << ftype << "\n";
    for (int k = 0; k < ZDIM; ++k)
        for (int j = 0; j < YDIM; ++j)
            for (int i = 0; i < XDIM; ++i) {
                int idx = k + j * ZDIM + i * YDIM * ZDIM; // Compute 1D index
                file << i << "," << j << "," << k << "," << x[idx] << "\n";
            }
    file.close();
}