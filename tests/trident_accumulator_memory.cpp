#include "matrix_blocks.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <omp.h>

int main(int argc, char** argv) {
    try {
        omp_set_num_threads(2);
        if (argc == 2) {
            const int n = std::stoi(argv[1]);
            if (n < 1 || n > 10000) throw std::runtime_error("invalid stress size");
            CsrMatrix a, b;
            a.rows = n; a.cols = 1;
            b.rows = 1; b.cols = n;
            for (int r = 0; r < n; ++r) {
                a.rowPtr.push_back(r);
                a.columnIndices.push_back(0);
                a.values.push_back(1.0);
                b.columnIndices.push_back(r);
                b.values.push_back(r + 1.0);
            }
            a.rowPtr.push_back(n);
            b.rowPtr = {0, n};
            trident::LocalAccumulator accumulator(n, n);
            accumulator.addProduct(a, b);
            accumulator.addProduct(a, b);
            const auto c = accumulator.finish();
            if (c.values.size() != static_cast<std::size_t>(n) * n)
                throw std::runtime_error("wrong dense output size");
            for (int r = 0; r < n; ++r)
                for (int j = 0; j < n; ++j) {
                    const auto p = c.rowPtr[r] + j;
                    if (c.columnIndices[p] != j || c.values[p] != 2.0 * (j + 1))
                        throw std::runtime_error("wrong dense output value");
                }
        } else {
            // Unsorted duplicates, cancellations across stages, and an empty row.
            CsrMatrix a, b;
            a.rows = 3; a.cols = 2;
            a.rowPtr = {0, 3, 3, 4}; a.columnIndices = {1, 0, 1, 0};
            a.values = {2.0, 1.0, -1.0}; a.values.push_back(3.0);
            b.rows = 2; b.cols = 4;
            b.rowPtr = {0, 3, 5}; b.columnIndices = {3, 1, 3, 0, 1};
            b.values = {2.0, 4.0, -1.0, 5.0, -4.0};
            trident::LocalAccumulator accumulator(3, 4);
            accumulator.addProduct(a, b);
            for (auto& v : a.values) v *= -1.0;
            accumulator.addProduct(a, b);
            for (auto& v : a.values) v *= -0.5;
            accumulator.addProduct(a, b);
            const auto c = accumulator.finish();
            const auto reference = serialSpgemm(a, b);
            if (maxAbsoluteDifference(c, reference) != 0.0 || c.rowPtr[1] != c.rowPtr[2])
                throw std::runtime_error("wrong multi-stage accumulation");
            trident::LocalAccumulator empty(0, 4);
            if (empty.finish().rowPtr != std::vector<CsrOffset>{0})
                throw std::runtime_error("wrong empty accumulator");
        }
        std::cout << "PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
