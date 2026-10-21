#include "spgemm_common.hpp"
#include "matrix_market.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

// Deliberately one process, no MPI_Init or product implementation under test.
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--reference-build-id") {
            std::cout << partialReferenceBuildId() << '\n';
            return 0;
        }
        const auto options = parseOptions(argc, argv, "prepare_partial_reference", "");
        if (options.partialCacheDirectory.empty() || options.matrixAPath.empty())
            throw std::invalid_argument("use --matrix-a A --matrix-b B --partial --partial-cache DIR");
        const auto start = std::chrono::steady_clock::now();
        const auto a = readMatrixMarket(options.matrixAPath);
        if (options.matrixBPath.empty() && a.rows != a.cols)
            throw std::invalid_argument("rectangular A requires --matrix-b");
        const auto b = options.matrixBPath.empty() ? a : readMatrixMarket(options.matrixBPath);
        std::cout << "validation_input_read_seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
        ValidationDiagnostics diagnostics;
        const auto reference = preparePartialReference(a, b, options.partialCacheDirectory, diagnostics);
        printValidationDiagnostics(diagnostics);
        std::cout << "validation_reference_finite=" << reference.finite << '\n';
        return reference.finite ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "partial reference error: " << e.what() << '\n';
        return 1;
    }
}
