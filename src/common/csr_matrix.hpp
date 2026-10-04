#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

inline int checkedCsrCount(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CSR storage and MPI counts must fit in a signed int");
    }
    return static_cast<int>(count);
}

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> rowPtr;
    std::vector<int> columnIndices;
    std::vector<double> values;
};
