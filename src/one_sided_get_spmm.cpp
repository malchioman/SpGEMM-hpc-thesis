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

constexpr char kProgramName[] = "spmm_one_sided_get";
constexpr char kImplementation[] = "mpi_openmp_one_sided_get";
constexpr char kDefaultResultsPath[] = "results/one_sided_get/benchmarks.tsv";

void fetchRemoteDenseRows(const std::vector<RowBlock>& denseBlocks, int denseCols,
                          MPI_Win denseWindow, RemoteDenseRows& remoteDenseRows,
                          MPI_Comm communicator) {
    for (int peer = 0; peer < static_cast<int>(denseBlocks.size()); ++peer) {
        const int peerOffset = remoteDenseRows.plan.peerOffsets[peer];
        const auto& rows = remoteDenseRows.plan.rowsByPeer[peer];
        for (int index = 0; index < static_cast<int>(rows.size()); ++index) {
            const int globalRow = rows[index];
            const MPI_Aint targetOffset =
                static_cast<MPI_Aint>(globalRow - denseBlocks[peer].firstRow) * denseCols;
            double* destination = remoteDenseRows.values.data() +
                                  static_cast<std::size_t>(peerOffset + index) * denseCols;
            checkMpi(MPI_Get(destination, denseCols, MPI_DOUBLE, peer, targetOffset, denseCols,
                             MPI_DOUBLE, denseWindow),
                     "MPI_Get(dense row)", communicator);
        }
    }
    checkMpi(MPI_Win_flush_all(denseWindow), "MPI_Win_flush_all", communicator);
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

    MPI_Win denseWindow = MPI_WIN_NULL;
    bool denseWindowLocked = false;
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
        RemoteDenseRows remoteDenseRows =
            makeRemoteDenseRows(buildRemoteRowPlan(localMatrix, denseBlocks, rank, communicator),
                                options.denseCols);
        checkMpi(
            MPI_Win_create(ownedDenseRows.empty() ? nullptr : ownedDenseRows.data(),
                           static_cast<MPI_Aint>(ownedDenseRows.size()) * sizeof(double),
                           sizeof(double), MPI_INFO_NULL, communicator, &denseWindow),
            "MPI_Win_create", communicator);
        checkMpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, denseWindow), "MPI_Win_lock_all", communicator);
        denseWindowLocked = true;
        // Publish local stores before remote ranks fetch from the exposed rows of B.
        checkMpi(MPI_Win_sync(denseWindow), "MPI_Win_sync", communicator);
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        fetchRemoteDenseRows(denseBlocks, options.denseCols, denseWindow, remoteDenseRows, communicator);
        const double haloSetupSeconds = maxElapsed(haloSetupStart, communicator);

        std::vector<double> localResult(static_cast<std::size_t>(localMatrix.rows) * options.denseCols);
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
            ++denseEpoch;
            fillOwnedDenseRows(denseBlocks[rank], options.denseCols, denseEpoch, ownedDenseRows);
            // Prevent the next get epoch from observing stale private copies of the window.
            checkMpi(MPI_Win_sync(denseWindow), "MPI_Win_sync", communicator);
            checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
            fetchRemoteDenseRows(denseBlocks, options.denseCols, denseWindow, remoteDenseRows,
                                 communicator);
            spmm(localMatrix, denseBlocks[rank], ownedDenseRows, remoteDenseRows, options.denseCols,
                 localResult);
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
                checkMpi(MPI_Win_sync(denseWindow), "MPI_Win_sync", communicator);
                const double start = MPI_Wtime();
                checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
                fetchRemoteDenseRows(denseBlocks, options.denseCols, denseWindow, remoteDenseRows,
                                     communicator);
                const double exchangeEnd = MPI_Wtime();
                spmm(localMatrix, denseBlocks[rank], ownedDenseRows, remoteDenseRows, options.denseCols,
                     localResult);
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
        if (denseWindow != MPI_WIN_NULL) {
            if (denseWindowLocked) {
                MPI_Win_unlock_all(denseWindow);
            }
            MPI_Win_free(&denseWindow);
        }
        fail(error.what(), communicator);
    }

    if (denseWindow != MPI_WIN_NULL) {
        if (denseWindowLocked) {
            checkMpi(MPI_Win_unlock_all(denseWindow), "MPI_Win_unlock_all", communicator);
        }
        checkMpi(MPI_Win_free(&denseWindow), "MPI_Win_free", communicator);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", communicator);
    return EXIT_SUCCESS;
}
