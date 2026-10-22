#include "spgemm_common.hpp"
#include "matrix_market.hpp"
#include "full_reference.hpp"

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
        const auto options = parseOptions(argc, argv, "prepare_full_reference", "");
        if (options.fullCacheDirectory.empty() || options.matrixAPath.empty())
            throw std::invalid_argument("use --matrix-a A --matrix-b B --full-cache DIR");
        const auto start = std::chrono::steady_clock::now();
        const auto a = readMatrixMarket(options.matrixAPath);
        if (options.matrixBPath.empty() && a.rows != a.cols)
            throw std::invalid_argument("rectangular A requires --matrix-b");
        const auto b = options.matrixBPath.empty() ? a : readMatrixMarket(options.matrixBPath);
        std::cout << "validation_input_read_seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
        ValidationDiagnostics diagnostics;
        prepareFullReference(a, b, options.fullCacheDirectory, diagnostics);
        printValidationDiagnostics(diagnostics);
        std::cout << "validation_reference_path=" << fullReferencePath(options.fullCacheDirectory, diagnostics.key)
                  << " validation_reference_bytes=" << std::filesystem::file_size(fullReferencePath(options.fullCacheDirectory, diagnostics.key)) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "full reference error: " << e.what() << '\n';
        return 1;
    }
}
