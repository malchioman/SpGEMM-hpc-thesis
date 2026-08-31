#include <mpi.h>
#include <omp.h>

#include "matrix_market.hpp"
#include "spmm_common.hpp"
#include "spmm_exchange.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr char kProgramName[] = "spmm_one_sided_put";
constexpr char kImplementation[] = "mpi_openmp_one_sided_put";
constexpr char kDefaultResultsPath[] = "results/one_sided_put/benchmarks.tsv";
constexpr int kPutRequestCountTag = 400;
constexpr int kPutRequestRowsTag = 401;

struct OutgoingPutPlan {
    std::vector<int> rows;
    std::vector<int> targetSlots;
};

struct PutPlan {
    RemoteDenseRows remoteDenseRows;
    std::vector<OutgoingPutPlan> outgoingByPeer;
};

template <typename T>
T* dataOrNull(std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

template <typename T>
const T* dataOrNull(const std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

PutPlan buildPutPlan(const CsrMatrix& localMatrix, const std::vector<RowBlock>& denseBlocks,
                     RowBlock localBlock, int denseCols, int rank, int ranks,
                     MPI_Comm communicator) {
    PutPlan plan;
    plan.remoteDenseRows =
        makeRemoteDenseRows(buildRemoteRowPlan(localMatrix, denseBlocks, rank, communicator),
                            denseCols);
    plan.outgoingByPeer.resize(ranks);

    std::vector<int> incomingCounts(ranks, 0);
    std::vector<int> outgoingCounts(ranks, 0);
    for (int peer = 0; peer < ranks; ++peer) {
        outgoingCounts[peer] = static_cast<int>(plan.remoteDenseRows.plan.rowsByPeer[peer].size());
    }

    std::vector<MPI_Request> requests;
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(&incomingCounts[peer], 1, MPI_INT, peer, kPutRequestCountTag,
                           communicator, &request),
                 "MPI_Irecv(put request count)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(&outgoingCounts[peer], 1, MPI_INT, peer, kPutRequestCountTag,
                           communicator, &request),
                 "MPI_Isend(put request count)", communicator);
        requests.push_back(request);
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

        const auto& rows = plan.remoteDenseRows.plan.rowsByPeer[peer];
        outgoingPacked[peer].resize(rows.size() * 2U);
        for (int index = 0; index < static_cast<int>(rows.size()); ++index) {
            outgoingPacked[peer][static_cast<std::size_t>(2 * index)] = rows[index];
            outgoingPacked[peer][static_cast<std::size_t>(2 * index + 1)] =
                plan.remoteDenseRows.plan.rowToSlot.at(rows[index]);
        }

        incomingPacked[peer].resize(static_cast<std::size_t>(incomingCounts[peer]) * 2U);
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(dataOrNull(incomingPacked[peer]),
                           static_cast<int>(incomingPacked[peer].size()), MPI_INT, peer,
                           kPutRequestRowsTag, communicator, &request),
                 "MPI_Irecv(put row requests)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(dataOrNull(outgoingPacked[peer]),
                           static_cast<int>(outgoingPacked[peer].size()), MPI_INT, peer,
                           kPutRequestRowsTag, communicator, &request),
                 "MPI_Isend(put row requests)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        const auto& packed = incomingPacked[peer];
        auto& outgoing = plan.outgoingByPeer[peer];
        const int rowCount = incomingCounts[peer];
        outgoing.rows.resize(rowCount);
        outgoing.targetSlots.resize(rowCount);
        for (int index = 0; index < rowCount; ++index) {
            const int globalRow = packed[static_cast<std::size_t>(2 * index)];
            const int targetSlot = packed[static_cast<std::size_t>(2 * index + 1)];
            if (globalRow < localBlock.firstRow || globalRow >= localBlock.firstRow + localBlock.rows) {
                fail("received an RMA put request for a non-local dense row", communicator);
            }
            outgoing.rows[index] = globalRow;
            outgoing.targetSlots[index] = targetSlot;
        }
    }

    return plan;
}

void pushDenseRows(const std::vector<OutgoingPutPlan>& outgoingByPeer, RowBlock localBlock,
                   const std::vector<double>& ownedDenseRows, int denseCols, MPI_Win haloWindow,
                   MPI_Comm communicator) {
    for (int peer = 0; peer < static_cast<int>(outgoingByPeer.size()); ++peer) {
        const auto& plan = outgoingByPeer[peer];
        for (int index = 0; index < static_cast<int>(plan.rows.size()); ++index) {
            const int localRow = plan.rows[index] - localBlock.firstRow;
            const double* source =
                ownedDenseRows.data() + static_cast<std::size_t>(localRow) * denseCols;
            const MPI_Aint targetOffset = static_cast<MPI_Aint>(plan.targetSlots[index]) * denseCols;
            checkMpi(MPI_Put(source, denseCols, MPI_DOUBLE, peer, targetOffset, denseCols, MPI_DOUBLE,
                             haloWindow),
                     "MPI_Put(dense row)", communicator);
        }
    }
    checkMpi(MPI_Win_flush_all(haloWindow), "MPI_Win_flush_all", communicator);
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

    MPI_Win haloWindow = MPI_WIN_NULL;
    bool haloWindowLocked = false;
    try {
        const Options options = parseOptions(argc, argv, kProgramName, kDefaultResultsPath);
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        omp_set_schedule(parseSchedule(options.schedule), options.chunk);

        CsrMatrix globalMatrix;
        std::array<int, 2> matrixShape{};
        if (rank == 0) {
            globalMatrix = options.matrixPath.empty()
                               ? makeBandedMatrix(options.rows, options.cols, options.nonZerosPerRow)
                               : readMatrixMarket(options.matrixPath);
            matrixShape = {globalMatrix.rows, globalMatrix.cols};
        }
        checkMpi(MPI_Bcast(matrixShape.data(), static_cast<int>(matrixShape.size()), MPI_INT, 0,
                           communicator),
                 "MPI_Bcast(matrix shape)", communicator);

        const std::vector<RowBlock> outputBlocks = partitionRows(matrixShape[0], ranks);
        const std::vector<RowBlock> denseBlocks = partitionRows(matrixShape[1], ranks);

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double distributionStart = MPI_Wtime();
        CsrMatrix localMatrix = distributeMatrix(rank == 0 ? &globalMatrix : nullptr, outputBlocks,
                                                 rank, ranks, communicator);
        const double distributionSeconds = maxElapsed(distributionStart, communicator);

        int denseEpoch = 0;
        std::vector<double> ownedDenseRows =
            makeOwnedDenseRows(denseBlocks[rank], options.denseCols, denseEpoch);

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double haloSetupStart = MPI_Wtime();
        PutPlan putPlan = buildPutPlan(localMatrix, denseBlocks, denseBlocks[rank], options.denseCols,
                                       rank, ranks, communicator);
        checkMpi(
            MPI_Win_create(putPlan.remoteDenseRows.values.empty() ? nullptr
                                                                  : putPlan.remoteDenseRows.values.data(),
                           static_cast<MPI_Aint>(putPlan.remoteDenseRows.values.size()) *
                               sizeof(double),
                           sizeof(double), MPI_INFO_NULL, communicator, &haloWindow),
            "MPI_Win_create", communicator);
        checkMpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, haloWindow), "MPI_Win_lock_all", communicator);
        haloWindowLocked = true;
        pushDenseRows(putPlan.outgoingByPeer, denseBlocks[rank], ownedDenseRows, options.denseCols,
                      haloWindow, communicator);
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        // Pull remote puts into the local private copy before the SpMM kernel reads the halo.
        checkMpi(MPI_Win_sync(haloWindow), "MPI_Win_sync", communicator);
        const double haloSetupSeconds = maxElapsed(haloSetupStart, communicator);

        std::vector<double> localResult(static_cast<std::size_t>(localMatrix.rows) * options.denseCols);
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
            ++denseEpoch;
            fillOwnedDenseRows(denseBlocks[rank], options.denseCols, denseEpoch, ownedDenseRows);
            pushDenseRows(putPlan.outgoingByPeer, denseBlocks[rank], ownedDenseRows, options.denseCols,
                          haloWindow, communicator);
            checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
            checkMpi(MPI_Win_sync(haloWindow), "MPI_Win_sync", communicator);
            spmm(localMatrix, denseBlocks[rank], ownedDenseRows, putPlan.remoteDenseRows,
                 options.denseCols, localResult);
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
                ++denseEpoch;
                fillOwnedDenseRows(denseBlocks[rank], options.denseCols, denseEpoch, ownedDenseRows);
                const double start = MPI_Wtime();
                pushDenseRows(putPlan.outgoingByPeer, denseBlocks[rank], ownedDenseRows, options.denseCols,
                              haloWindow, communicator);
                checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
                checkMpi(MPI_Win_sync(haloWindow), "MPI_Win_sync", communicator);
                const double exchangeEnd = MPI_Wtime();
                spmm(localMatrix, denseBlocks[rank], ownedDenseRows, putPlan.remoteDenseRows,
                     options.denseCols, localResult);
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
        const std::vector<double> globalResult =
            gatherResult(localResult, outputBlocks, options.denseCols, rank, ranks, communicator);
        const double gatherSeconds = maxElapsed(gatherStart, communicator);

        if (rank == 0) {
            const std::vector<double> reference = serialSpmm(globalMatrix, options.denseCols, denseEpoch);
            const double error = maxAbsoluteDifference(globalResult, reference);
            const double floatingPointOperations = 2.0 * globalMatrix.values.size() * options.denseCols;
            const double computeGflops = floatingPointOperations / computeP90Seconds / 1.0e9;
            appendBenchmarkResult(kImplementation, options, ranks, globalMatrix, distributionSeconds,
                                  haloSetupSeconds, communicationP90Seconds, computeP90Seconds,
                                  endToEndP90Seconds, gatherSeconds, computeGflops, error);
            printBenchmarkSummary(kImplementation, options, ranks, globalMatrix, distributionSeconds,
                                  haloSetupSeconds, communicationP90Seconds, computeP90Seconds,
                                  endToEndP90Seconds, gatherSeconds, computeGflops, error);
        }
    } catch (const std::exception& error) {
        if (haloWindow != MPI_WIN_NULL) {
            if (haloWindowLocked) {
                MPI_Win_unlock_all(haloWindow);
            }
            MPI_Win_free(&haloWindow);
        }
        fail(error.what(), communicator);
    }

    if (haloWindow != MPI_WIN_NULL) {
        if (haloWindowLocked) {
            checkMpi(MPI_Win_unlock_all(haloWindow), "MPI_Win_unlock_all", communicator);
        }
        checkMpi(MPI_Win_free(&haloWindow), "MPI_Win_free", communicator);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", communicator);
    return EXIT_SUCCESS;
}
