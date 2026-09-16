#include "spgemm_common.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

int main() {
    const CsrMatrix matrixA{2, 3, {0, 2, 3}, {0, 2, 1}, {1.0, 2.0, 3.0}};
    const CsrMatrix matrixB{3, 2, {0, 1, 2, 3}, {1, 0, 1}, {4.0, 5.0, 6.0}};
    const CsrMatrix expected{2, 2, {0, 1, 2}, {1, 0}, {16.0, 15.0}};

    const CsrMatrix actual = serialSpgemm(matrixA, matrixB);
    const double error = maxAbsoluteDifference(actual, expected);
    if (!std::isfinite(error) || error > 1.0e-12) {
        std::cerr << "unexpected serial SpGEMM result, max error=" << error << '\n';
        return 1;
    }

    const std::int64_t multiplications = scalarMultiplicationCount(matrixA, matrixB);
    if (multiplications != 3) {
        std::cerr << "unexpected scalar multiplication count=" << multiplications << '\n';
        return 1;
    }

    const CsrMatrix zero{1, 1, {0, 0}, {}, {}};
    const CsrMatrix finite{1, 1, {0, 1}, {0}, {2.0}};
    for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity()}) {
        const CsrMatrix nonFinite{1, 1, {0, 1}, {0}, {value}};
        for (const CsrMatrix& other : {zero, finite, nonFinite}) {
            if (validationPassed(maxAbsoluteDifference(nonFinite, other)) ||
                validationPassed(maxAbsoluteDifference(other, nonFinite))) {
                std::cerr << "non-finite matrix accepted by validation\n";
                return 1;
            }
        }
        if (validationPassed(value)) {
            std::cerr << "non-finite error accepted by validation\n";
            return 1;
        }
    }
    if (!validationPassed(0.0) || !validationPassed(1.0e-12) ||
        validationPassed(1.0e-10) || validationPassed(-1.0) ||
        maxAbsoluteDifference(zero, finite) != 2.0 ||
        maxAbsoluteDifference(finite, zero) != 2.0) {
        std::cerr << "unexpected validation threshold or sparse comparison\n";
        return 1;
    }
    return 0;
}
