#include "matrix_blocks.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace trident {
namespace {

void sendMatrix(const CsrMatrix& matrix, int target, MPI_Comm comm) {
    const std::array<int, 3> header{matrix.rows, matrix.cols, mpiCount(matrix.values.size())};
    checkMpi(MPI_Send(header.data(), 3, MPI_INT, target, 10, comm), "MPI_Send(block header)", comm);
    checkMpi(MPI_Send(matrix.rowPtr.data(), mpiCount(matrix.rowPtr.size()), MPI_INT, target, 11, comm),
             "MPI_Send(block pointers)", comm);
    checkMpi(MPI_Send(matrix.columnIndices.data(), header[2], MPI_INT, target, 12, comm),
             "MPI_Send(block indices)", comm);
    checkMpi(MPI_Send(matrix.values.data(), header[2], MPI_DOUBLE, target, 13, comm),
             "MPI_Send(block values)", comm);
}

CsrMatrix receiveMatrix(int source, MPI_Comm comm) {
    std::array<int, 3> header{};
    checkMpi(MPI_Recv(header.data(), 3, MPI_INT, source, 10, comm, MPI_STATUS_IGNORE),
             "MPI_Recv(block header)", comm);
    CsrMatrix matrix;
    matrix.rows = header[0];
    matrix.cols = header[1];
    matrix.rowPtr.resize(static_cast<std::size_t>(matrix.rows) + 1);
    matrix.columnIndices.resize(header[2]);
    matrix.values.resize(header[2]);
    checkMpi(MPI_Recv(matrix.rowPtr.data(), mpiCount(matrix.rowPtr.size()), MPI_INT, source, 11,
                       comm, MPI_STATUS_IGNORE), "MPI_Recv(block pointers)", comm);
    checkMpi(MPI_Recv(matrix.columnIndices.data(), header[2], MPI_INT, source, 12, comm,
                       MPI_STATUS_IGNORE), "MPI_Recv(block indices)", comm);
    checkMpi(MPI_Recv(matrix.values.data(), header[2], MPI_DOUBLE, source, 13, comm, MPI_STATUS_IGNORE),
             "MPI_Recv(block values)", comm);
    return matrix;
}

}  // namespace

int mpiCount(std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Trident CSR storage and MPI counts must fit in a signed int");
    }
    return static_cast<int>(size);
}

MatrixBlock matrixBlock(int rows, int cols, const ProcessGrid& grid, int i, int j, int k) {
    const auto coarseRows = partitionRows(rows, grid.side).at(i);
    const auto localRows = partitionRows(coarseRows.rows, grid.nodeSize).at(k);
    return {{coarseRows.firstRow + localRows.firstRow, localRows.rows},
            partitionRows(cols, grid.side).at(j)};
}

CsrMatrix sliceBlock(const CsrMatrix& matrix, MatrixBlock block) {
    CsrMatrix result;
    result.rows = block.rows.rows;
    result.cols = block.cols.rows;
    result.rowPtr.push_back(0);
    for (int row = block.rows.firstRow; row < block.rows.firstRow + block.rows.rows; ++row) {
        for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
            const int col = matrix.columnIndices[p] - block.cols.firstRow;
            if (col >= 0 && col < result.cols) {
                result.columnIndices.push_back(col);
                result.values.push_back(matrix.values[p]);
            }
        }
        result.rowPtr.push_back(mpiCount(result.values.size()));
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
                    for (int p = matrix.rowPtr[r]; p < matrix.rowPtr[r + 1]; ++p) {
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
        result.rowPtr.push_back(mpiCount(result.values.size()));
    }
    return result;
}

LocalAccumulator::LocalAccumulator(int rows, int cols) : cols_(cols), rows_(rows) {}

void LocalAccumulator::addProduct(const CsrMatrix& a, const CsrMatrix& b) {
    if (a.cols != b.rows || a.rows != mpiCount(rows_.size()) || b.cols != cols_) {
        throw std::invalid_argument("incompatible Trident partial product shapes");
    }
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < a.rows; ++row) {
        auto& accumulator = rows_[row];
        for (int p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p) {
            const int bRow = a.columnIndices[p];
            for (int t = b.rowPtr[bRow]; t < b.rowPtr[bRow + 1]; ++t) {
                accumulator[b.columnIndices[t]] += a.values[p] * b.values[t];
            }
        }
    }
}

CsrMatrix LocalAccumulator::finish() const {
    const int rowCount = mpiCount(rows_.size());
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
        result.rowPtr.push_back(mpiCount(static_cast<std::size_t>(result.rowPtr.back()) + row.size()));
    }
    result.columnIndices.resize(result.rowPtr.back());
    result.values.resize(result.rowPtr.back());
    #pragma omp parallel for schedule(runtime)
    for (int row = 0; row < rowCount; ++row) {
        int p = result.rowPtr[row];
        for (const auto& entry : sortedRows[row]) {
            result.columnIndices[p] = entry.first;
            result.values[p++] = entry.second;
        }
    }
    return result;
}

}  // namespace trident
