#pragma once
#include <vector>

class Grid{
    std::vector<double> data;
    void fill(double value) {
        std::fill(data.begin(), data.end(), value);
    }
};