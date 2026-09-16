#pragma once

#include <vector>

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> rowPtr;
    std::vector<int> columnIndices;
    std::vector<double> values;
};
