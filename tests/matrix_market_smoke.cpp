#include "matrix_market.hpp"

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: matrix_market_smoke <matrix.mtx>\n";
        return 2;
    }

    const CsrMatrix matrix = readMatrixMarket(argv[1]);
    if (matrix.rows != 3 || matrix.cols != 3 || matrix.values.size() != 6 ||
        matrix.rowPtr != std::vector<int>({0, 2, 4, 6})) {
        std::cerr << "unexpected CSR conversion\n";
        return 1;
    }
    return 0;
}
