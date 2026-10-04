#include "spgemm_common.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

CsrMatrix input(int rows, int cols, int salt) {
    CsrMatrix result{rows, cols, {0}, {}, {}};
    for (int row = 0; row < rows; ++row) {
        // Deliberately unsorted, duplicate coordinates, explicit zeros and empty rows.
        for (int entry = 0; cols && (row + salt) % 4 && entry < 2 * cols + 1; ++entry) {
            result.columnIndices.push_back((entry * 7 + row + salt) % cols);
            result.values.push_back((entry * 3 + row + salt) % 7 - 3);
        }
        result.rowPtr.push_back(static_cast<int>(result.values.size()));
    }
    return result;
}

std::vector<double> dense(const CsrMatrix& matrix) {
    std::vector<double> values(static_cast<std::size_t>(matrix.rows) * matrix.cols, 0);
    for (int row = 0; row < matrix.rows; ++row) {
        for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
            values[static_cast<std::size_t>(row) * matrix.cols + matrix.columnIndices[p]] += matrix.values[p];
        }
    }
    return values;
}

bool checkDenseOracle() {
    for (int seed = 0; seed < 100; ++seed) {
        const auto a = input(seed % 8, (seed * 3) % 7, seed);
        const auto b = input(a.cols, (seed * 5) % 9, seed + 1);
        const auto product = serialSpgemm(a, b);
        const auto da = dense(a), db = dense(b), dc = dense(product);
        if (product.rows != a.rows || product.cols != b.cols) return false;
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < b.cols; ++j) {
                double expected = 0;
                for (int k = 0; k < a.cols; ++k) expected += da[i * a.cols + k] * db[k * b.cols + j];
                if (dc[i * b.cols + j] != expected) return false;
            }
            int previous = -1;
            for (int p = product.rowPtr[i]; p < product.rowPtr[i + 1]; ++p) {
                if (product.columnIndices[p] <= previous || product.values[p] == 0) return false;
                previous = product.columnIndices[p];
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    if (!checkDenseOracle()) {
        std::cerr << "serial SpGEMM disagrees with independent dense oracle\n";
        return 1;
    }
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (checkedCsrCount(limit) != std::numeric_limits<int>::max()) return 1;
    try {
        checkedCsrCount(limit + 1);
        std::cerr << "CSR count overflow accepted\n";
        return 1;
    } catch (const std::overflow_error&) {}
    try {
        // Must reject before allocating billions of entries.
        makeBandedMatrix(1 << 30, 2, 2);
        return 1;
    } catch (const std::overflow_error&) {}
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
