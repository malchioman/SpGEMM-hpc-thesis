#include "intra_node_exchange.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

CsrMatrix sparseInput(int rows, int cols, int salt) {
    CsrMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.rowPtr.push_back(0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if ((i + salt) % 5 != 0 && (i * 13 + j * 7 + salt) % 4 != 0) {
                const int value = (i * 3 + j * 11 + salt) % 9 - 4;
                if (value != 0) {
                    result.columnIndices.push_back(j);
                    result.values.push_back(value);
                }
            }
        }
        result.rowPtr.push_back(static_cast<int>(result.values.size()));
    }
    return result;
}

std::vector<double> dense(const CsrMatrix& matrix) {
    std::vector<double> result(static_cast<std::size_t>(matrix.rows) * matrix.cols, 0.0);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int p = matrix.rowPtr[i]; p < matrix.rowPtr[i + 1]; ++p) {
            result[static_cast<std::size_t>(i) * matrix.cols + matrix.columnIndices[p]] += matrix.values[p];
        }
    }
    return result;
}

void checkProduct(const CsrMatrix& a, const CsrMatrix& b, const CsrMatrix& actual) {
    if (actual.rows != a.rows || actual.cols != b.cols || actual.rowPtr.size() != static_cast<std::size_t>(a.rows) + 1) {
        throw std::runtime_error("wrong result dimensions");
    }
    if (actual.rowPtr.front() != 0 || actual.rowPtr.back() != static_cast<int>(actual.values.size()) ||
        actual.columnIndices.size() != actual.values.size()) {
        throw std::runtime_error("invalid result CSR");
    }
    for (int i = 0; i < actual.rows; ++i) {
        int previous = -1;
        for (int p = actual.rowPtr[i]; p < actual.rowPtr[i + 1]; ++p) {
            if (actual.columnIndices[p] <= previous || actual.columnIndices[p] >= actual.cols || actual.values[p] == 0.0) {
                throw std::runtime_error("result contains unsorted/duplicate columns or stored zeros");
            }
            previous = actual.columnIndices[p];
        }
    }
    const auto da = dense(a);
    const auto db = dense(b);
    const auto dc = dense(actual);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < b.cols; ++j) {
            double expected = 0.0;
            for (int k = 0; k < a.cols; ++k) {
                expected += da[static_cast<std::size_t>(i) * a.cols + k] * db[static_cast<std::size_t>(k) * b.cols + j];
            }
            if (dc[static_cast<std::size_t>(i) * b.cols + j] != expected) {
                throw std::runtime_error("Trident disagrees with independent dense product");
            }
        }
    }
}

void runCase(CsrMatrix globalA, CsrMatrix globalB, const trident::ProcessGrid& grid) {
    auto a = trident::distributeBlocks(&globalA, globalA.rows, globalA.cols, grid);
    auto b = trident::distributeBlocks(&globalB, globalB.rows, globalB.cols, grid);
    const auto plan = trident::preparePlan(a, b, grid);
    trident::ProductWorkspace workspace(plan);
    trident::TwoSidedIntraNodeExchange exchange(grid);
    // Reuse the plan after changing values and indices (all tile sizes stay fixed).
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto product = trident::multiply(a, b, grid, plan, exchange, workspace);
        const auto actual = trident::gatherBlocks(product.matrix, globalA.rows, globalB.cols, grid);
        if (grid.rank == 0) checkProduct(globalA, globalB, actual);
        for (double& value : b.values) value *= -2.0;
        for (double& value : globalB.values) value *= -2.0;
        // Rotate within each coarse column block to preserve ownership and message sizes.
        for (int& col : b.columnIndices) col = (col + 1) % b.cols;
        const auto colBlocks = partitionRows(globalB.cols, grid.side);
        for (int& col : globalB.columnIndices) {
            const auto block = colBlocks[ownerOfRow(col, colBlocks)];
            col = block.firstRow + (col - block.firstRow + 1) % block.rows;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return EXIT_FAILURE;
    if (provided < MPI_THREAD_FUNNELED) fail("insufficient MPI thread level", MPI_COMM_WORLD);
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(2);
        omp_set_schedule(omp_sched_dynamic, 1);
        // Reverse world ranks to exercise mappings independently of MPI_COMM_WORLD numbering.
        int worldRank = 0;
        checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank", MPI_COMM_WORLD);
        MPI_Comm reversed = MPI_COMM_NULL;
        checkMpi(MPI_Comm_split(MPI_COMM_WORLD, 0, -worldRank, &reversed), "MPI_Comm_split", MPI_COMM_WORLD);
        trident::ProcessGrid grid(reversed, argc > 1 ? std::stoi(argv[1]) : 0);
        if (argc > 2 && std::string(argv[2]) == "large") {
            // Payloads exceed common eager thresholds, exercising matched rendezvous transfers.
            runCase(sparseInput(385, 389, 1), sparseInput(389, 383, 2), grid);
        }
        for (const auto shape : {std::array<int, 3>{7, 9, 5}, {1, 1, 1}, {0, 3, 2}, {3, 0, 4}, {3, 4, 0}, {19, 11, 13}}) {
            runCase(sparseInput(shape[0], shape[1], 1), sparseInput(shape[1], shape[2], 2), grid);
        }
        runCase({2, 2, {0, 2, 4}, {0, 1, 0, 1}, {1, 1, 1, -1}},
                {2, 2, {0, 2, 4}, {0, 1, 0, 1}, {1, -1, -1, 1}}, grid);
        runCase({3, 4, {0, 0, 0, 0}, {}, {}}, sparseInput(4, 2, 3), grid);
        if (grid.rank == 0) std::cout << "Trident dense-oracle, repeated products, empty tiles and cancellation: PASS\n";
        grid.close();
        checkMpi(MPI_Comm_free(&reversed), "MPI_Comm_free", MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fail(error.what(), MPI_COMM_WORLD);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}
