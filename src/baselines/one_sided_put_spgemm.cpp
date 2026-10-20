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

constexpr char kProgramName[] = "spgemm_one_sided_put";
constexpr char kImplementation[] = "mpi_openmp_one_sided_put";
constexpr char kDefaultResultsPath[] = "results/one_sided_put/benchmarks_v2.tsv";
constexpr int kPutRequestCountTag = 400;
constexpr int kPutRequestRowsTag = 401;
constexpr int kPutRowNnzTag = 402;
constexpr int kPutTargetOffsetTag = 403;

struct OutgoingPutPlan {
    std::vector<int> rows;
    std::vector<CsrOffset> targetOffsets;
};

struct PutPlan {
    RemoteSparseRows remoteBRows;
    std::vector<OutgoingPutPlan> outgoingByPeer;
};

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

void unlockAndFree(MPI_Win& window, bool locked, MPI_Comm communicator) {
    if (window == MPI_WIN_NULL) {
        return;
    }
    if (locked) {
        checkMpi(MPI_Win_unlock_all(window), "MPI_Win_unlock_all", communicator);
    }
    checkMpi(MPI_Win_free(&window), "MPI_Win_free", communicator);
}

CsrOffset localRowNonZeros(const CsrMatrix& matrix, int localRow) {
    return matrix.rowPtr[localRow + 1] - matrix.rowPtr[localRow];
}

PutPlan buildPutPlan(const CsrMatrix& localMatrixA, const CsrMatrix& localMatrixB,
                     const std::vector<RowBlock>& bBlocks, RowBlock localBBlock, int rank,
                     int ranks, MPI_Comm communicator) {
    PutPlan plan;
    plan.remoteBRows =
        makeRemoteSparseRows(buildRemoteRowPlan(localMatrixA, bBlocks, rank, communicator));
    plan.outgoingByPeer.resize(ranks);

    std::vector<int> incomingCounts(ranks, 0);
    std::vector<int> outgoingCounts(ranks, 0);
    for (int peer = 0; peer < ranks; ++peer) {
        outgoingCounts[peer] = checkedMatrixDimension(checkedCsrCount(plan.remoteBRows.plan.rowsByPeer[peer].size()));
    }

    std::vector<MPI_Request> requests;
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        csrIrecv(&incomingCounts[peer], 1, peer, kPutRequestCountTag, communicator, requests);
        csrIsend(&outgoingCounts[peer], 1, peer, kPutRequestCountTag, communicator, requests);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<int>> incomingPacked(ranks);
    std::vector<std::vector<int>> outgoingPacked(ranks);
    requests.clear();
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        const auto& rows = plan.remoteBRows.plan.rowsByPeer[peer];
        outgoingPacked[peer].resize(rows.size() * 2U);
        for (int index = 0; index < checkedMatrixDimension(checkedCsrCount(rows.size())); ++index) {
            outgoingPacked[peer][(2 * static_cast<std::size_t>(index))] = rows[index];
            outgoingPacked[peer][(2 * static_cast<std::size_t>(index) + 1)] =
                plan.remoteBRows.plan.rowToSlot.at(rows[index]);
        }

        incomingPacked[peer].resize(static_cast<std::size_t>(incomingCounts[peer]) * 2U);

        csrIrecv(dataOrNull(incomingPacked[peer]), checkedCsrCount(incomingPacked[peer].size()), peer, kPutRequestRowsTag, communicator, requests);
        csrIsend(dataOrNull(outgoingPacked[peer]), checkedCsrCount(outgoingPacked[peer].size()), peer, kPutRequestRowsTag, communicator, requests);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<CsrOffset>> incomingRowNonZeros(ranks);
    std::vector<std::vector<CsrOffset>> outgoingRowNonZeros(ranks);
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        incomingRowNonZeros[peer].resize(outgoingCounts[peer]);
        outgoingRowNonZeros[peer].resize(incomingCounts[peer]);
        for (int index = 0; index < incomingCounts[peer]; ++index) {
            const int globalRow = incomingPacked[peer][(2 * static_cast<std::size_t>(index))];
            if (globalRow < localBBlock.firstRow ||
                globalRow >= localBBlock.firstRow + localBBlock.rows) {
                fail("received an RMA put request for a non-local sparse row of B", communicator);
            }
            outgoingRowNonZeros[peer][index] =
                localRowNonZeros(localMatrixB, globalRow - localBBlock.firstRow);
        }
    }

    requests.clear();
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        csrIrecv(dataOrNull(incomingRowNonZeros[peer]), outgoingCounts[peer], peer, kPutRowNnzTag, communicator, requests);
        csrIsend(dataOrNull(outgoingRowNonZeros[peer]), checkedMatrixDimension(checkedCsrCount(outgoingRowNonZeros[peer].size())), peer, kPutRowNnzTag, communicator, requests);
    }
    waitAll(requests, communicator);

    CsrOffset totalRemoteNonZeros = 0;
    for (int peer = 0; peer < ranks; ++peer) {
        const int peerOffset = plan.remoteBRows.plan.peerOffsets[peer];
        for (int index = 0; index < checkedMatrixDimension(checkedCsrCount(incomingRowNonZeros[peer].size())); ++index) {
            plan.remoteBRows.rowPtr[peerOffset + index] = totalRemoteNonZeros;
            totalRemoteNonZeros = checkedCsrAdd(totalRemoteNonZeros, incomingRowNonZeros[peer][index], "halo prefix");
            plan.remoteBRows.rowPtr[peerOffset + index + 1] = totalRemoteNonZeros;
        }
    }
    plan.remoteBRows.columnIndices.assign(csrSize<int>(totalRemoteNonZeros), 0);
    plan.remoteBRows.values.assign(csrSize<double>(totalRemoteNonZeros), 0.0);

    std::vector<std::vector<CsrOffset>> incomingTargetOffsets(ranks);
    std::vector<std::vector<CsrOffset>> outgoingTargetOffsets(ranks);
    requests.clear();
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        outgoingTargetOffsets[peer].resize(outgoingCounts[peer]);
        const auto& rows = plan.remoteBRows.plan.rowsByPeer[peer];
        for (int index = 0; index < checkedMatrixDimension(checkedCsrCount(rows.size())); ++index) {
            const int slot = plan.remoteBRows.plan.rowToSlot.at(rows[index]);
            outgoingTargetOffsets[peer][index] = plan.remoteBRows.rowPtr[slot];
        }
        incomingTargetOffsets[peer].resize(incomingCounts[peer]);

        csrIrecv(dataOrNull(incomingTargetOffsets[peer]), incomingCounts[peer], peer, kPutTargetOffsetTag, communicator, requests);
        csrIsend(dataOrNull(outgoingTargetOffsets[peer]), checkedMatrixDimension(checkedCsrCount(outgoingTargetOffsets[peer].size())), peer, kPutTargetOffsetTag, communicator, requests);
    }
    waitAll(requests, communicator);

    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        auto& outgoing = plan.outgoingByPeer[peer];
        outgoing.rows.resize(incomingCounts[peer]);
        outgoing.targetOffsets.resize(incomingCounts[peer]);
        for (int index = 0; index < incomingCounts[peer]; ++index) {
            const int globalRow = incomingPacked[peer][(2 * static_cast<std::size_t>(index))];
            const CsrOffset targetOffset = incomingTargetOffsets[peer][index];
            if (globalRow < localBBlock.firstRow ||
                globalRow >= localBBlock.firstRow + localBBlock.rows) {
                fail("received an RMA put request for a non-local sparse row of B", communicator);
            }
            if (targetOffset < 0) {
                fail("received a negative RMA put target offset", communicator);
            }
            outgoing.rows[index] = globalRow;
            outgoing.targetOffsets[index] = targetOffset;
        }
    }

    return plan;
}

void pushSparseRows(const std::vector<OutgoingPutPlan>& outgoingByPeer, RowBlock localBBlock,
                    const CsrMatrix& localMatrixB, MPI_Win columnWindow, MPI_Win valueWindow,
                    MPI_Comm communicator) {
    for (int peer = 0; peer < checkedMatrixDimension(checkedCsrCount(outgoingByPeer.size())); ++peer) {
        const auto& outgoing = outgoingByPeer[peer];
        for (int index = 0; index < checkedMatrixDimension(checkedCsrCount(outgoing.rows.size())); ++index) {
            const int localRow = outgoing.rows[index] - localBBlock.firstRow;
            const CsrOffset first = localMatrixB.rowPtr[localRow];
            const CsrOffset count = localMatrixB.rowPtr[localRow + 1] - first;
            if (count == 0) {
                continue;
            }

            const CsrOffset targetOffset = outgoing.targetOffsets[index];
            csrPut(localMatrixB.columnIndices.data() + first, count, peer, targetOffset, columnWindow, communicator);
            csrPut(localMatrixB.values.data() + first, count, peer, targetOffset, valueWindow, communicator);
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
    PutPlan putPlan;
    MPI_Win columnWindow = MPI_WIN_NULL;
    MPI_Win valueWindow = MPI_WIN_NULL;
    bool windowsLocked = false;
    int exitCode = EXIT_SUCCESS;

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
        checkMpi(MPI_Bcast(matrixShape.data(), checkedMatrixDimension(checkedCsrCount(matrixShape.size())), MPI_INT, 0,
                           communicator),
                 "MPI_Bcast(matrix shape)", communicator);

        const std::vector<RowBlock> outputBlocks = partitionRows(matrixShape[0], ranks);
        const std::vector<RowBlock> bRowBlocks = partitionRows(matrixShape[2], ranks);

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double distributionStart = MPI_Wtime();
        CsrMatrix localMatrixA = distributeMatrix(rank == 0 ? &globalMatrixA : nullptr, outputBlocks,
                                                  rank, ranks, communicator);
        CsrMatrix localMatrixB = distributeMatrix(rank == 0 ? &globalMatrixB : nullptr, bRowBlocks,
                                                  rank, ranks, communicator);
        // Defer reporting reductions until the complete first product has finished.
        const double distributionElapsed = MPI_Wtime() - distributionStart;

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double haloSetupStart = MPI_Wtime();
        putPlan = buildPutPlan(localMatrixA, localMatrixB, bRowBlocks, bRowBlocks[rank],
                               rank, ranks, communicator);
        checkMpi(MPI_Win_create(dataOrNull(putPlan.remoteBRows.columnIndices),
                                csrWindowBytes<int>(putPlan.remoteBRows.columnIndices.size()),
                                sizeof(int), MPI_INFO_NULL, communicator, &columnWindow),
                 "MPI_Win_create(columns)", communicator);
        checkMpi(MPI_Win_create(dataOrNull(putPlan.remoteBRows.values),
                                csrWindowBytes<double>(putPlan.remoteBRows.values.size()),
                                sizeof(double), MPI_INFO_NULL, communicator, &valueWindow),
                 "MPI_Win_create(values)", communicator);
        checkMpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, columnWindow), "MPI_Win_lock_all(columns)",
                 communicator);
        checkMpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, valueWindow), "MPI_Win_lock_all(values)",
                 communicator);
        windowsLocked = true;
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double haloSetupEnd = MPI_Wtime();
        pushSparseRows(putPlan.outgoingByPeer, bRowBlocks[rank], localMatrixB, columnWindow,
                       valueWindow, communicator);
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        checkMpi(MPI_Win_sync(columnWindow), "MPI_Win_sync(columns)", communicator);
        checkMpi(MPI_Win_sync(valueWindow), "MPI_Win_sync(values)", communicator);
        CsrMatrix localResult =
            spgemm(localMatrixA, bRowBlocks[rank], localMatrixB, putPlan.remoteBRows, communicator);
        const double firstProductEnd = MPI_Wtime();
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier(first gather)", communicator);
        const double firstGatherStart = MPI_Wtime();
        double firstGatherEnd = 0.0;
        {
            const auto firstResult = gatherCsrMatrix(
                localResult, outputBlocks, matrixShape[3], rank, ranks, communicator);
            firstGatherEnd = MPI_Wtime();
            (void)firstResult;  // Release the collected first product before warmup.
        }
        const double fullProductSeconds = maxRankValue(firstGatherEnd - distributionStart, communicator);
        const double firstGatherSeconds = maxRankValue(firstGatherEnd - firstGatherStart, communicator);
        const double distributionSeconds = maxRankValue(distributionElapsed, communicator);
        const double haloSetupSeconds = maxRankValue(haloSetupEnd - haloSetupStart, communicator);
        const double firstProductSeconds = maxRankValue(firstProductEnd - haloSetupStart, communicator);

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
            pushSparseRows(putPlan.outgoingByPeer, bRowBlocks[rank], localMatrixB, columnWindow,
                           valueWindow, communicator);
            checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
            checkMpi(MPI_Win_sync(columnWindow), "MPI_Win_sync(columns)", communicator);
            checkMpi(MPI_Win_sync(valueWindow), "MPI_Win_sync(values)", communicator);
            localResult =
                spgemm(localMatrixA, bRowBlocks[rank], localMatrixB, putPlan.remoteBRows,
                       communicator);
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
                pushSparseRows(putPlan.outgoingByPeer, bRowBlocks[rank], localMatrixB, columnWindow,
                               valueWindow, communicator);
                checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
                checkMpi(MPI_Win_sync(columnWindow), "MPI_Win_sync(columns)", communicator);
                checkMpi(MPI_Win_sync(valueWindow), "MPI_Win_sync(values)", communicator);
                const double exchangeEnd = MPI_Wtime();
                localResult =
                    spgemm(localMatrixA, bRowBlocks[rank], localMatrixB, putPlan.remoteBRows,
                           communicator);
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
        const auto work = rank == 0 ? rowPartitionWork(globalMatrixA, globalMatrixB, ranks)
                                    : std::vector<std::int64_t>{};
        const auto metrics = collectBenchmarkMetrics(
            globalMatrixA, globalMatrixB, globalResult, localMatrixA, localMatrixB, localResult,
            work, communicator);

        if (rank == 0) {
            const auto validation = validateProduct(globalMatrixA, globalMatrixB, globalResult,
                                                     options.validationMode);
            exitCode = validation.passed ? EXIT_SUCCESS : EXIT_FAILURE;
            const double floatingPointOperations =
                2.0 * static_cast<double>(scalarMultiplicationCount(globalMatrixA, globalMatrixB));
            const double computeGflops = computeP90Seconds > 0.0
                                             ? floatingPointOperations / computeP90Seconds / 1.0e9
                                             : 0.0;
            appendBenchmarkResult(kImplementation, options, ranks, globalMatrixA, globalMatrixB,
                                  globalResult, distributionSeconds, haloSetupSeconds, firstProductSeconds,
                                  communicationP90Seconds, computeP90Seconds, endToEndP90Seconds,
                                  gatherSeconds, computeGflops, validation,
                                  communicationSamples, computeSamples, endToEndSamples,
                                  fullProductSeconds, firstGatherSeconds, metrics);
            printBenchmarkSummary(kImplementation, options, ranks, globalMatrixA, globalMatrixB,
                                  globalResult, distributionSeconds, haloSetupSeconds, firstProductSeconds,
                                  communicationP90Seconds, computeP90Seconds, endToEndP90Seconds,
                                  gatherSeconds, computeGflops, validation,
                                  fullProductSeconds, firstGatherSeconds);
        }
        checkMpi(MPI_Bcast(&exitCode, 1, MPI_INT, 0, communicator), "MPI_Bcast(validation)", communicator);
    } catch (const std::exception& error) {
        // Abort directly: other ranks may still be in communication or setup.
        fail(error.what(), communicator);
    }

    unlockAndFree(valueWindow, windowsLocked, communicator);
    unlockAndFree(columnWindow, windowsLocked, communicator);
    checkMpi(MPI_Finalize(), "MPI_Finalize", communicator);
    return exitCode;
}
