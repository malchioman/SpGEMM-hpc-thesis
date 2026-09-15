#include "matrix_market.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: matrix_market_shape_smoke <matrix.mtx> <rectangular|reject-nonsquare>\n";
        return 2;
    }
    const std::string mode = argv[2];
    if (mode != "rectangular" && mode != "reject-nonsquare") {
        std::cerr << "unknown test mode: " << mode << '\n';
        return 2;
    }
    const bool expectRejection = mode == "reject-nonsquare";

    try {
        const CsrMatrix matrix = readMatrixMarket(argv[1]);
        if (expectRejection) {
            std::cerr << "accepted a nonsquare matrix with a symmetry storage mode\n";
            return 1;
        }
        const CsrMatrix expected{2, 3, {0, 1, 2}, {2, 0}, {5.0, 7.0}};
        if (matrix.rows != expected.rows || matrix.cols != expected.cols ||
            matrix.rowPtr != expected.rowPtr || matrix.columnIndices != expected.columnIndices ||
            matrix.values != expected.values) {
            std::cerr << "unexpected CSR conversion of a general rectangular matrix\n";
            return 1;
        }
    } catch (const std::runtime_error& error) {
        if (expectRejection &&
            std::string(error.what()).find("requires a square matrix") != std::string::npos) {
            return 0;
        }
        std::cerr << "unexpected reader error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
