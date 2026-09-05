#include <mpi.h>
#include <omp.h>

#include "matrix_market.hpp"
#include "spgemm_common.hpp"
#include "spgemm_exchange.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kProgramName[] = "spgemm_two_sided";
constexpr char kImplementation[] = "mpi_openmp_two_sided";
constexpr char kDefaultResultsPath[] = "results/two_sided/benchmarks.tsv";

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
        CsrMatrix localMatrixB = distributeMatrix(rank == 0 ? &globalMatrixB : nullptr, bRowBlocks,
                                                  rank, ranks, communicator);
        const double distributionSeconds = maxElapsed(distributionStart, communicator);

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double haloSetupStart = MPI_Wtime();
        RemoteSparseRows remoteBRows = exchangeRemoteSparseRowsTwoSided(
            localMatrixA, localMatrixB, bRowBlocks, bRowBlocks[rank], rank, ranks, communicator);
        const double haloSetupSeconds = maxElapsed(haloSetupStart, communicator);

        CsrMatrix localResult;
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            remoteBRows = exchangeRemoteSparseRowsTwoSided(localMatrixA, localMatrixB, bRowBlocks,
                                                           bRowBlocks[rank], rank, ranks,
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

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        for (int trial = 0; trial < options.trials; ++trial) {
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
                const double start = MPI_Wtime();
                remoteBRows = exchangeRemoteSparseRowsTwoSided(localMatrixA, localMatrixB, bRowBlocks,
                                                               bRowBlocks[rank], rank, ranks,
                                                               communicator);
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
            const CsrMatrix reference = serialSpgemm(globalMatrixA, globalMatrixB);
            const double error = maxAbsoluteDifference(globalResult, reference);
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
        fail(error.what(), communicator);
    }

    checkMpi(MPI_Finalize(), "MPI_Finalize", communicator);
    return EXIT_SUCCESS;
}
