#include "matrix_blocks.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace trident {
namespace {

void sendMatrix(const CsrMatrix& matrix, int target, MPI_Comm comm) {
    const std::array<CsrOffset, 3> header{matrix.rows, matrix.cols, checkedCsrCount(matrix.values.size())};
    csrSend(header.data(), 3, target, 10, comm);
    csrSend(matrix.rowPtr.data(), checkedCsrCount(matrix.rowPtr.size()), target, 11, comm);
    csrSend(matrix.columnIndices.data(), header[2], target, 12, comm);
    csrSend(matrix.values.data(), header[2], target, 13, comm);
}

CsrMatrix receiveMatrix(int source, MPI_Comm comm) {
    std::array<CsrOffset, 3> header{};
    csrRecv(header.data(), 3, source, 10, comm);
    CsrMatrix matrix;
    matrix.rows = checkedMatrixDimension(header[0]);
    matrix.cols = checkedMatrixDimension(header[1]);
    matrix.rowPtr.resize(static_cast<std::size_t>(matrix.rows) + 1);
    matrix.columnIndices.resize(csrSize<int>(header[2]));
    matrix.values.resize(csrSize<double>(header[2]));
    csrRecv(matrix.rowPtr.data(), checkedCsrCount(matrix.rowPtr.size()), source, 11, comm);
    csrRecv(matrix.columnIndices.data(), header[2], source, 12, comm);
    csrRecv(matrix.values.data(), header[2], source, 13, comm);
    return matrix;
}

}  // namespace

MatrixBlock matrixBlock(int rows, int cols, const ProcessGrid& grid, int i, int j, int k) {
    const auto coarseRows = partitionRows(rows, grid.side).at(i);
    const auto localRows = partitionRows(coarseRows.rows, grid.nodeSize).at(k);
    return {{coarseRows.firstRow + localRows.firstRow, localRows.rows},
            partitionRows(cols, grid.side).at(j)};
}

std::vector<std::int64_t> scalarProductsByRank(const CsrMatrix& a, const CsrMatrix& b,
                                             const ProcessGrid& grid) {
    std::vector<std::int64_t> work(grid.size, 0);
    // One column tile at a time limits temporary storage to O(B.rows + ranks).
    for (int j = 0; j < grid.side; ++j) {
        const auto columns = partitionRows(b.cols, grid.side)[j];
        std::vector<CsrOffset> rowDegrees(b.rows);
        for (int row = 0; row < b.rows; ++row) {
            const auto begin = b.columnIndices.begin() + b.rowPtr[row];
            const auto end = b.columnIndices.begin() + b.rowPtr[row + 1];
            // Matrix Market, permutation and synthetic inputs can have unsorted columns.
            // Count stored entries exactly as the kernel does, including duplicates/zeros.
            rowDegrees[row] = static_cast<CsrOffset>(std::count_if(begin, end, [&](int column) {
                return column >= columns.firstRow && column < columns.firstRow + columns.rows;
            }));
        }
        for (int i = 0; i < grid.side; ++i) {
            for (int k = 0; k < grid.nodeSize; ++k) {
                const auto rows = matrixBlock(a.rows, b.cols, grid, i, j, k).rows;
                auto& count = work[grid.rankAt(i, j, k)];
                for (int row = rows.firstRow; row < rows.firstRow + rows.rows; ++row) {
                    for (CsrOffset p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p) {
                        count = checkedCsrAdd(count, rowDegrees[a.columnIndices[p]], "Trident scalar products");
                    }
                }
            }
        }
    }
    return work;
}

CsrMatrix sliceBlock(const CsrMatrix& matrix, MatrixBlock block) {
    CsrMatrix result;
    result.rows = block.rows.rows;
    result.cols = block.cols.rows;
    result.rowPtr.push_back(0);
    for (int row = block.rows.firstRow; row < block.rows.firstRow + block.rows.rows; ++row) {
        for (CsrOffset p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
            const int col = matrix.columnIndices[p] - block.cols.firstRow;
            if (col >= 0 && col < result.cols) {
                result.columnIndices.push_back(col);
                result.values.push_back(matrix.values[p]);
            }
        }
        result.rowPtr.push_back(checkedCsrCount(result.values.size()));
    }
    return result;
}

CsrMatrix distributeBlocks(const CsrMatrix* matrix, int rows, int cols, const ProcessGrid& grid) {
    if (grid.rank != 0) {
        return receiveMatrix(0, grid.world);
    }
    CsrMatrix local;
    for (int i = 0; i < grid.side; ++i) {
        for (int j = 0; j < grid.side; ++j) {
            for (int k = 0; k < grid.nodeSize; ++k) {
                auto block = sliceBlock(*matrix, matrixBlock(rows, cols, grid, i, j, k));
                const int target = grid.rankAt(i, j, k);
                if (target == 0) {
                    local = std::move(block);
                } else {
                    sendMatrix(block, target, grid.world);
                }
            }
        }
    }
    return local;
}

CsrMatrix gatherBlocks(const CsrMatrix& local, int rows, int cols, const ProcessGrid& grid) {
    if (grid.rank != 0) {
        sendMatrix(local, 0, grid.world);
        return {};
    }
    std::vector<std::vector<std::pair<int, double>>> entries(rows);
    for (int i = 0; i < grid.side; ++i) {
        for (int j = 0; j < grid.side; ++j) {
            for (int k = 0; k < grid.nodeSize; ++k) {
                const int source = grid.rankAt(i, j, k);
                const auto block = matrixBlock(rows, cols, grid, i, j, k);
                const auto matrix = source == 0 ? local : receiveMatrix(source, grid.world);
                if (matrix.rows != block.rows.rows || matrix.cols != block.cols.rows) {
                    throw std::runtime_error("unexpected Trident result block shape");
                }
                for (int r = 0; r < matrix.rows; ++r) {
                    auto& row = entries[block.rows.firstRow + r];
                    for (CsrOffset p = matrix.rowPtr[r]; p < matrix.rowPtr[r + 1]; ++p) {
                        row.emplace_back(block.cols.firstRow + matrix.columnIndices[p], matrix.values[p]);
                    }
                }
            }
        }
    }
    CsrMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.rowPtr.push_back(0);
    for (auto& row : entries) {
        std::sort(row.begin(), row.end());
        for (const auto& entry : row) {
            result.columnIndices.push_back(entry.first);
            result.values.push_back(entry.second);
        }
        result.rowPtr.push_back(checkedCsrCount(result.values.size()));
    }
    return result;
}

LocalAccumulator::LocalAccumulator(int rows, int cols) : cols_(cols), rows_(rows) {}

void LocalAccumulator::addProduct(const CsrMatrix& a, const CsrMatrix& b) {
    if (a.cols != b.rows || a.rows != checkedCsrCount(rows_.size()) || b.cols != cols_) {
        throw std::invalid_argument("incompatible Trident partial product shapes");
    }
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < a.rows; ++row) {
        auto& accumulator = rows_[row];
        for (CsrOffset p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p) {
            const int bRow = a.columnIndices[p];
            for (CsrOffset t = b.rowPtr[bRow]; t < b.rowPtr[bRow + 1]; ++t) {
                accumulator[b.columnIndices[t]] += a.values[p] * b.values[t];
            }
        }
    }
}

CsrMatrix LocalAccumulator::finish() const {
    const int rowCount = checkedMatrixDimension(checkedCsrCount(rows_.size()));
    std::vector<std::vector<std::pair<int, double>>> sortedRows(rowCount);
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < rowCount; ++row) {
        for (const auto& entry : rows_[row]) {
            if (entry.second != 0.0) {
                sortedRows[row].push_back(entry);
            }
        }
        std::sort(sortedRows[row].begin(), sortedRows[row].end());
    }
    CsrMatrix result;
    result.rows = rowCount;
    result.cols = cols_;
    result.rowPtr.push_back(0);
    for (const auto& row : sortedRows) {
        result.rowPtr.push_back(checkedCsrAdd(result.rowPtr.back(), checkedCsrCount(row.size()), "Trident local result prefix"));
    }
    result.columnIndices.resize(csrSize<int>(result.rowPtr.back()));
    result.values.resize(csrSize<double>(result.rowPtr.back()));
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < rowCount; ++row) {
        CsrOffset p = result.rowPtr[row];
        for (const auto& entry : sortedRows[row]) {
            result.columnIndices[p] = entry.first;
            result.values[p++] = entry.second;
        }
    }
    return result;
}

}  // namespace trident
