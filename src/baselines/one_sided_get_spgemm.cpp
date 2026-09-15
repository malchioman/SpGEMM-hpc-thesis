#include <mpi.h>
#include <omp.h>

#include "matrix_market.hpp"
#include "spgemm_common.hpp"
#include "spgemm_exchange.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kProgramName[] = "spgemm_one_sided_get";
constexpr char kImplementation[] = "mpi_openmp_one_sided_get";
constexpr char kDefaultResultsPath[] = "results/one_sided_get/benchmarks.tsv";

CsrMatrix loadMatrixA(const Options& options) {
    if (!options.matrixAPath.empty()) {
        return readMatrixMarket(options.matrixAPath);
    }
    return makeBandedMatrix(options.rows, options.cols, options.nonZerosPerRow);
}

CsrMatrix loadMatrixB(const Options& options, const CsrMatrix& matrixA) {
    if (!options.matrixBPath.empty()) {
        return readMatrixMarket(options.matrixBPath);
    }
    if (!options.matrixAPath.empty()) {
        if (matrixA.rows != matrixA.cols) {
            throw std::invalid_argument(
                "--matrix-a without --matrix-b requires a square A so it can be reused as B");
        }
        return matrixA;
    }
    return makeBandedMatrix(options.cols, options.bCols, options.bNonZerosPerRow);
}

void validateDimensions(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    if (matrixA.cols != matrixB.rows) {
        throw std::invalid_argument("SpGEMM requires A.cols == B.rows");
    }
}

template <typename T>
T* dataOrNull(std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

void unlockAndFree(MPI_Win& window, bool locked) {
    if (window == MPI_WIN_NULL) {
        return;
    }
    if (locked) {
        MPI_Win_unlock_all(window);
    }
    MPI_Win_free(&window);
}

void fetchRemoteSparseRows(const std::vector<RowBlock>& bBlocks, MPI_Win rowPtrWindow,
                           MPI_Win columnWindow, MPI_Win valueWindow,
                           RemoteSparseRows& remoteRows, MPI_Comm communicator) {
    std::vector<int> rowBounds(remoteRows.plan.flatRows.size() * 2U, 0);
    for (int peer = 0; peer < static_cast<int>(bBlocks.size()); ++peer) {
        const int peerOffset = remoteRows.plan.peerOffsets[peer];
        const auto& rows = remoteRows.plan.rowsByPeer[peer];
        for (int index = 0; index < static_cast<int>(rows.size()); ++index) {
            const int globalRow = rows[index];
            const int localRow = globalRow - bBlocks[peer].firstRow;
            const int slot = peerOffset + index;
            checkMpi(MPI_Get(rowBounds.data() + static_cast<std::size_t>(2 * slot), 2, MPI_INT,
                             peer, static_cast<MPI_Aint>(localRow), 2, MPI_INT, rowPtrWindow),
                     "MPI_Get(B row bounds)", communicator);
        }
    }
    checkMpi(MPI_Win_flush_all(rowPtrWindow), "MPI_Win_flush_all(rowPtr)", communicator);

    int totalRemoteNonZeros = 0;
    remoteRows.rowPtr.assign(remoteRows.plan.flatRows.size() + 1, 0);
    for (int slot = 0; slot < static_cast<int>(remoteRows.plan.flatRows.size()); ++slot) {
        const int first = rowBounds[static_cast<std::size_t>(2 * slot)];
        const int last = rowBounds[static_cast<std::size_t>(2 * slot + 1)];
        if (first < 0 || last < first) {
            fail("received invalid CSR row bounds from a remote B row", communicator);
        }
        remoteRows.rowPtr[slot] = totalRemoteNonZeros;
        totalRemoteNonZeros += last - first;
        remoteRows.rowPtr[slot + 1] = totalRemoteNonZeros;
    }
    remoteRows.columnIndices.assign(totalRemoteNonZeros, 0);
    remoteRows.values.assign(totalRemoteNonZeros, 0.0);

    for (int peer = 0; peer < static_cast<int>(bBlocks.size()); ++peer) {
        const int peerOffset = remoteRows.plan.peerOffsets[peer];
        const auto& rows = remoteRows.plan.rowsByPeer[peer];
        for (int index = 0; index < static_cast<int>(rows.size()); ++index) {
            const int slot = peerOffset + index;
            const int receiveOffset = remoteRows.rowPtr[slot];
            const int count = remoteRows.rowPtr[slot + 1] - remoteRows.rowPtr[slot];
            if (count == 0) {
                continue;
            }

            const int remoteOffset = rowBounds[static_cast<std::size_t>(2 * slot)];
            checkMpi(MPI_Get(remoteRows.columnIndices.data() + receiveOffset, count, MPI_INT, peer,
                             static_cast<MPI_Aint>(remoteOffset), count, MPI_INT, columnWindow),
                     "MPI_Get(B row columns)", communicator);
            checkMpi(MPI_Get(remoteRows.values.data() + receiveOffset, count, MPI_DOUBLE, peer,
                             static_cast<MPI_Aint>(remoteOffset), count, MPI_DOUBLE, valueWindow),
                     "MPI_Get(B row values)", communicator);
        }
    }
    checkMpi(MPI_Win_flush_all(columnWindow), "MPI_Win_flush_all(columns)", communicator);
    checkMpi(MPI_Win_flush_all(valueWindow), "MPI_Win_flush_all(values)", communicator);
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed\n";
        return EXIT_FAILURE;
    }

    MPI_Comm communicator = MPI_COMM_WORLD;
    int rank = 0;
    int ranks = 0;
    checkMpi(MPI_Comm_rank(communicator, &rank), "MPI_Comm_rank", communicator);
    checkMpi(MPI_Comm_size(communicator, &ranks), "MPI_Comm_size", communicator);
    if (provided < MPI_THREAD_FUNNELED) {
        fail("MPI implementation does not provide MPI_THREAD_FUNNELED", communicator);
    }

    // Keep exposed storage alive through window cleanup, including exception handling.
    CsrMatrix localMatrixB;
    MPI_Win rowPtrWindow = MPI_WIN_NULL;
    MPI_Win columnWindow = MPI_WIN_NULL;
    MPI_Win valueWindow = MPI_WIN_NULL;
    bool windowsLocked = false;
    int validationSuccess = 0;

    try {
        const Options options = parseOptions(argc, argv, kProgramName, kDefaultResultsPath);
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        omp_set_schedule(parseSchedule(options.schedule), options.chunk);

        CsrMatrix globalMatrixA;
        CsrMatrix globalMatrixB;
        std::array<int, 4> matrixShape{};
        if (rank == 0) {
            globalMatrixA = loadMatrixA(options);
            globalMatrixB = loadMatrixB(options, globalMatrixA);
            validateDimensions(globalMatrixA, globalMatrixB);
            matrixShape = {globalMatrixA.rows, globalMatrixA.cols, globalMatrixB.rows,
                           globalMatrixB.cols};
        }
        checkMpi(MPI_Bcast(matrixShape.data(), static_cast<int>(matrixShape.size()), MPI_INT, 0,
                           communicator),
                 "MPI_Bcast(matrix shape)", communicator);

        const std::vector<RowBlock> outputBlocks = partitionRows(matrixShape[0], ranks);
        const std::vector<RowBlock> bRowBlocks = partitionRows(matrixShape[2], ranks);

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double distributionStart = MPI_Wtime();
        CsrMatrix localMatrixA = distributeMatrix(rank == 0 ? &globalMatrixA : nullptr, outputBlocks,
                                                  rank, ranks, communicator);
        localMatrixB = distributeMatrix(rank == 0 ? &globalMatrixB : nullptr, bRowBlocks,
                                        rank, ranks, communicator);
        const double distributionSeconds = maxElapsed(distributionStart, communicator);

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double haloSetupStart = MPI_Wtime();
        RemoteSparseRows remoteBRows =
            makeRemoteSparseRows(buildRemoteRowPlan(localMatrixA, bRowBlocks, rank, communicator));
        checkMpi(MPI_Win_create(dataOrNull(localMatrixB.rowPtr),
                                static_cast<MPI_Aint>(localMatrixB.rowPtr.size()) * sizeof(int),
                                sizeof(int), MPI_INFO_NULL, communicator, &rowPtrWindow),
                 "MPI_Win_create(rowPtr)", communicator);
        checkMpi(MPI_Win_create(dataOrNull(localMatrixB.columnIndices),
                                static_cast<MPI_Aint>(localMatrixB.columnIndices.size()) *
                                    sizeof(int),
                                sizeof(int), MPI_INFO_NULL, communicator, &columnWindow),
                 "MPI_Win_create(columns)", communicator);
        checkMpi(MPI_Win_create(dataOrNull(localMatrixB.values),
                                static_cast<MPI_Aint>(localMatrixB.values.size()) * sizeof(double),
                                sizeof(double), MPI_INFO_NULL, communicator, &valueWindow),
                 "MPI_Win_create(values)", communicator);
        checkMpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, rowPtrWindow), "MPI_Win_lock_all(rowPtr)",
                 communicator);
        checkMpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, columnWindow), "MPI_Win_lock_all(columns)",
                 communicator);
        checkMpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, valueWindow), "MPI_Win_lock_all(values)",
                 communicator);
        windowsLocked = true;
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        fetchRemoteSparseRows(bRowBlocks, rowPtrWindow, columnWindow, valueWindow, remoteBRows,
                              communicator);
        const double haloSetupSeconds = maxElapsed(haloSetupStart, communicator);

        CsrMatrix localResult;
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
            fetchRemoteSparseRows(bRowBlocks, rowPtrWindow, columnWindow, valueWindow, remoteBRows,
                                  communicator);
            localResult =
                spgemm(localMatrixA, bRowBlocks[rank], localMatrixB, remoteBRows, communicator);
        }

        std::vector<double> computeSamples;
        std::vector<double> communicationSamples;
        std::vector<double> endToEndSamples;
        if (rank == 0) {
            const std::size_t sampleCount =
                static_cast<std::size_t>(options.repeats) * options.trials;
            computeSamples.reserve(sampleCount);
            communicationSamples.reserve(sampleCount);
            endToEndSamples.reserve(sampleCount);
        }

        for (int trial = 0; trial < options.trials; ++trial) {
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
                const double start = MPI_Wtime();
                fetchRemoteSparseRows(bRowBlocks, rowPtrWindow, columnWindow, valueWindow,
                                      remoteBRows, communicator);
                const double exchangeEnd = MPI_Wtime();
                localResult =
                    spgemm(localMatrixA, bRowBlocks[rank], localMatrixB, remoteBRows, communicator);
                const double end = MPI_Wtime();
                const double localTimings[3] = {exchangeEnd - start, end - exchangeEnd, end - start};
                double maxTimings[3] = {};
                checkMpi(MPI_Reduce(localTimings, maxTimings, 3, MPI_DOUBLE, MPI_MAX, 0,
                                    communicator),
                         "MPI_Reduce(sample timings)", communicator);
                if (rank == 0) {
                    communicationSamples.push_back(maxTimings[0]);
                    computeSamples.push_back(maxTimings[1]);
                    endToEndSamples.push_back(maxTimings[2]);
                }
            }
        }

        const double computeP90Seconds = rank == 0 ? percentile90(computeSamples) : 0.0;
        const double communicationP90Seconds = rank == 0 ? percentile90(communicationSamples) : 0.0;
        const double endToEndP90Seconds = rank == 0 ? percentile90(endToEndSamples) : 0.0;

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double gatherStart = MPI_Wtime();
        const CsrMatrix globalResult =
            gatherCsrMatrix(localResult, outputBlocks, matrixShape[3], rank, ranks, communicator);
        const double gatherSeconds = maxElapsed(gatherStart, communicator);

        if (rank == 0) {
            std::optional<double> error;
            if (options.validate) {
                const CsrMatrix reference = serialSpgemm(globalMatrixA, globalMatrixB);
                error = maxAbsoluteDifference(globalResult, reference);
            }
            validationSuccess = (!error || validationPassed(*error)) ? 1 : 0;
            const double floatingPointOperations =
                2.0 * static_cast<double>(scalarMultiplicationCount(globalMatrixA, globalMatrixB));
            const double computeGflops = computeP90Seconds > 0.0
                                             ? floatingPointOperations / computeP90Seconds / 1.0e9
                                             : 0.0;
            appendBenchmarkResult(kImplementation, options, ranks, globalMatrixA, globalMatrixB,
                                  globalResult, distributionSeconds, haloSetupSeconds,
                                  communicationP90Seconds, computeP90Seconds, endToEndP90Seconds,
                                  gatherSeconds, computeGflops, error);
            printBenchmarkSummary(kImplementation, options, ranks, globalMatrixA, globalMatrixB,
                                  globalResult, distributionSeconds, haloSetupSeconds,
                                  communicationP90Seconds, computeP90Seconds, endToEndP90Seconds,
                                  gatherSeconds, computeGflops, error);
        }
    } catch (const std::exception& error) {
        unlockAndFree(valueWindow, windowsLocked);
        unlockAndFree(columnWindow, windowsLocked);
        unlockAndFree(rowPtrWindow, windowsLocked);
        fail(error.what(), communicator);
    }

    unlockAndFree(valueWindow, windowsLocked);
    unlockAndFree(columnWindow, windowsLocked);
    unlockAndFree(rowPtrWindow, windowsLocked);
    checkMpi(MPI_Bcast(&validationSuccess, 1, MPI_INT, 0, communicator),
             "MPI_Bcast(validation status)", communicator);
    checkMpi(MPI_Finalize(), "MPI_Finalize", communicator);
    return validationSuccess ? EXIT_SUCCESS : EXIT_FAILURE;
}
