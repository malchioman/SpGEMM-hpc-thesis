#include "spgemm_common.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    const CsrMatrix matrixA{2, 3, {0, 2, 3}, {0, 2, 1}, {1.0, 2.0, 3.0}};
    const CsrMatrix matrixB{3, 2, {0, 1, 2, 3}, {1, 0, 1}, {4.0, 5.0, 6.0}};
    const CsrMatrix expected{2, 2, {0, 1, 2}, {1, 0}, {16.0, 15.0}};

    const CsrMatrix actual = serialSpgemm(matrixA, matrixB);
    const double error = maxAbsoluteDifference(actual, expected);
    if (error > 1.0e-12) {
        std::cerr << "unexpected serial SpGEMM result, max error=" << error << '\n';
        return 1;
    }

    const std::int64_t multiplications = scalarMultiplicationCount(matrixA, matrixB);
    if (multiplications != 3) {
        std::cerr << "unexpected scalar multiplication count=" << multiplications << '\n';
        return 1;
    }

    return 0;
}
