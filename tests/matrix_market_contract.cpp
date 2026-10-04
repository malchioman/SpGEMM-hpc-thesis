#include "matrix_market.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path path(argv[1]);
    const std::string header = "%%MatrixMarket matrix coordinate real general\n";
    const std::vector<std::string> invalid{
        header + "2 2 1\n1 1 1\n2 2 3\n",  // previously silently truncated
        header + "2 2 1\n1 1 1 junk\n", header + "2 2 1 extra\n1 1 1\n",
        header + "2 2 1\n1 1 nan\n", header + "2 2 1\n1 1 1e999\n",
        header + "2 2 1\n-2147483648 1 1\n", header + "2 2 1\n",
        "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n1 1 2\n"};
    for (const auto& content : invalid) {
        { std::ofstream out(path); out << content; }
        try {
            readMatrixMarket(path.string());
            std::cerr << "invalid input accepted: " << content;
            std::filesystem::remove(path);
            return 1;
        } catch (const std::runtime_error&) {}
    }
    { std::ofstream out(path); out << header << "1 2 3\n1 2 1D0\n1 1 0\n1 2 -1\n% trailing comment\n"; }
    const auto matrix = readMatrixMarket(path.string());
    std::filesystem::remove(path);
    // The input representation is intentionally preserved, not canonicalized.
    if (matrix.rowPtr != std::vector<int>{0, 3} || matrix.columnIndices != std::vector<int>{1, 0, 1} ||
        matrix.values != std::vector<double>{1, 0, -1}) return 1;
    return 0;
}
