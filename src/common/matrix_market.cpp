#include "matrix_market.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool isCommentOrEmpty(const std::string& line) {
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    return first == std::string::npos || line[first] == '%';
}

}  // namespace

CsrMatrix readMatrixMarket(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open Matrix Market file: " + path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Matrix Market file is empty: " + path);
    }
    std::istringstream header(line);
    std::string banner;
    std::string object;
    std::string storage;
    std::string field;
    std::string symmetry;
    header >> banner >> object >> storage >> field >> symmetry;
    object = lowercase(object);
    storage = lowercase(storage);
    field = lowercase(field);
    symmetry = lowercase(symmetry);
    if (banner != "%%MatrixMarket" || object != "matrix" || storage != "coordinate") {
        throw std::runtime_error("only Matrix Market coordinate matrices are supported: " + path);
    }
    if (field != "real" && field != "integer" && field != "pattern") {
        throw std::runtime_error("only real, integer, and pattern Matrix Market fields are supported: " + path);
    }
    if (symmetry != "general" && symmetry != "symmetric" && symmetry != "skew-symmetric" &&
        symmetry != "hermitian") {
        throw std::runtime_error("unsupported Matrix Market symmetry: " + symmetry);
    }

    do {
        if (!std::getline(input, line)) {
            throw std::runtime_error("Matrix Market file has no size line: " + path);
        }
    } while (isCommentOrEmpty(line));

    int rows = 0;
    int cols = 0;
    int entries = 0;
    {
        std::istringstream dimensions(line);
        if (!(dimensions >> rows >> cols >> entries) || rows <= 0 || cols <= 0 || entries < 0) {
            throw std::runtime_error("invalid Matrix Market dimensions: " + path);
        }
    }

    const bool mirrored = symmetry != "general";
    if (mirrored && rows != cols) {
        throw std::runtime_error("Matrix Market " + symmetry +
                                 " storage requires a square matrix: " + path);
    }

    struct Entry {
        int row;
        int column;
        double value;
    };
    std::vector<Entry> entriesList;
    entriesList.reserve(static_cast<std::size_t>(entries) * (mirrored ? 2U : 1U));
    for (int entry = 0; entry < entries; ++entry) {
        do {
            if (!std::getline(input, line)) {
                throw std::runtime_error("unexpected end of Matrix Market entries: " + path);
            }
        } while (isCommentOrEmpty(line));

        std::replace(line.begin(), line.end(), 'D', 'E');
        std::replace(line.begin(), line.end(), 'd', 'e');
        std::istringstream values(line);
        int row = 0;
        int column = 0;
        double value = 1.0;
        if (!(values >> row >> column) || (field != "pattern" && !(values >> value))) {
            throw std::runtime_error("invalid Matrix Market entry: " + path);
        }
        --row;
        --column;
        if (row < 0 || row >= rows || column < 0 || column >= cols) {
            throw std::runtime_error("Matrix Market entry index is out of range: " + path);
        }
        entriesList.push_back({row, column, value});
        if (mirrored && row != column) {
            entriesList.push_back({column, row, symmetry == "skew-symmetric" ? -value : value});
        }
    }

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.rowPtr.assign(rows + 1, 0);
    for (const Entry& entry : entriesList) {
        ++matrix.rowPtr[entry.row + 1];
    }
    for (int row = 0; row < rows; ++row) {
        matrix.rowPtr[row + 1] += matrix.rowPtr[row];
    }
    matrix.columnIndices.resize(entriesList.size());
    matrix.values.resize(entriesList.size());
    std::vector<int> cursor = matrix.rowPtr;
    for (const Entry& entry : entriesList) {
        const int position = cursor[entry.row]++;
        matrix.columnIndices[position] = entry.column;
        matrix.values[position] = entry.value;
    }
    return matrix;
}
