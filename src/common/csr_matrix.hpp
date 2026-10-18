#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using CsrOffset = std::int64_t;
static_assert(sizeof(int) == 4, "CSR column indices require 32-bit int");

inline CsrOffset checkedCsrCount(std::size_t count) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<CsrOffset>::max())) {
        throw std::overflow_error("CSR element count exceeds INT64_MAX: " + std::to_string(count));
    }
    return static_cast<CsrOffset>(count);
}

inline CsrOffset checkedCsrAdd(CsrOffset a, CsrOffset b, const char* context = "CSR prefix") {
    if (a < 0 || b < 0 || a > std::numeric_limits<CsrOffset>::max() - b) {
        throw std::overflow_error(std::string(context) + ": invalid/overflowing count " +
                                  std::to_string(a) + " + " + std::to_string(b));
    }
    return a + b;
}

inline int checkedMatrixDimension(CsrOffset value) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::overflow_error("matrix dimension exceeds signed int: " + std::to_string(value));
    }
    return static_cast<int>(value);
}

template <typename T>
std::size_t csrSize(CsrOffset count) {
    if (count < 0 || static_cast<std::uint64_t>(count) > std::vector<T>().max_size()) {
        throw std::overflow_error("CSR allocation is not addressable: " + std::to_string(count));
    }
    return static_cast<std::size_t>(count);
}

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<CsrOffset> rowPtr;
    std::vector<int> columnIndices;
    std::vector<double> values;
};
