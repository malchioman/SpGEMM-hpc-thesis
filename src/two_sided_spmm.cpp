#include <mpi.h>
#include <omp.h>

#include "csr_matrix.hpp"
#include "matrix_market.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kHeaderTag = 100;
constexpr int kRowPtrTag = 101;
constexpr int kColumnTag = 102;
constexpr int kValueTag = 103;
constexpr int kRequestCountTag = 200;
constexpr int kRequestRowsTag = 201;
constexpr int kDenseRowsTag = 202;
constexpr int kResultTag = 300;

struct Options {
    int rows = 1'024;
    int cols = 1'024;
    int denseCols = 32;
    int nonZerosPerRow = 16;
    int threads = 1;
    int warmup = 2;
    int repeats = 10;
    int trials = 5;
    int chunk = 64;
    std::string schedule = "guided";
    std::string matrixPath;
    std::string resultsPath = "results/two_sided/benchmarks.tsv";
    std::string experiment = "manual";
};

struct RowBlock {
    int firstRow = 0;
    int rows = 0;
};

[[noreturn]] void fail(const std::string& message, MPI_Comm communicator) {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::cerr << "rank " << rank << ": " << message << '\n';
    MPI_Abort(communicator, EXIT_FAILURE);
    std::abort();
}

void checkMpi(int error, const char* call, MPI_Comm communicator) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char errorString[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(error, errorString, &length);
    fail(std::string(call) + " failed: " + std::string(errorString, length), communicator);
}

void printUsage() {
    std::cout
        << "Usage: spmm_two_sided [options]\n"
        << "  --rows N          Rows of sparse matrix A (default: 1024)\n"
        << "  --cols N          Columns of sparse matrix A / rows of B (default: 1024)\n"
        << "  --dense-cols N    Columns of dense matrix B (default: 32)\n"
        << "  --nnz-per-row N   Non-zeros in each row of A (default: 16)\n"
        << "  --threads N       OpenMP threads per MPI rank (default: 1)\n"
        << "  --warmup N        Untimed SpMM repetitions before measurement (default: 2)\n"
        << "  --repeats N       Timed repetitions in each trial (default: 10)\n"
        << "  --trials N        Number of independent trials (default: 5)\n"
        << "  --matrix PATH     Matrix Market coordinate input; synthetic matrix if omitted\n"
        << "  --results PATH    TSV output path (default: results/two_sided/benchmarks.tsv)\n"
        << "  --experiment TAG  Experiment label stored in the TSV (default: manual)\n"
        << "  --schedule NAME   OpenMP schedule: static, dynamic, guided, auto (default: guided)\n"
        << "  --chunk N         OpenMP schedule chunk, ignored by auto (default: 64)\n";
}

int parsePositiveInt(const char* value, const char* option) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
            throw std::invalid_argument("not positive");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects a positive integer");
    }
}

int parseNonNegativeInt(const char* value, const char* option) {
    try {
        const int parsed = std::stoi(value);
        if (parsed < 0) {
            throw std::invalid_argument("negative");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects a non-negative integer");
    }
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            printUsage();
            std::exit(EXIT_SUCCESS);
        }
        if (index + 1 == argc) {
            throw std::invalid_argument("missing value for " + argument);
        }

        if (argument == "--schedule" || argument == "--matrix" || argument == "--results" ||
            argument == "--experiment") {
            const std::string value = argv[++index];
            if (argument == "--matrix") {
                options.matrixPath = value;
                continue;
            }
            if (argument == "--results") {
                options.resultsPath = value;
                continue;
            }
            if (argument == "--experiment") {
                options.experiment = value;
                continue;
            }
            options.schedule = value;
            if (options.schedule != "static" && options.schedule != "dynamic" &&
                options.schedule != "guided" && options.schedule != "auto") {
                throw std::invalid_argument("--schedule expects static, dynamic, guided, or auto");
            }
            continue;
        }

        const char* rawValue = argv[++index];
        const int value = argument == "--warmup"
                              ? parseNonNegativeInt(rawValue, argument.c_str())
                              : parsePositiveInt(rawValue, argument.c_str());
        if (argument == "--rows") {
            options.rows = value;
        } else if (argument == "--cols") {
            options.cols = value;
        } else if (argument == "--dense-cols") {
            options.denseCols = value;
        } else if (argument == "--nnz-per-row") {
            options.nonZerosPerRow = value;
        } else if (argument == "--threads") {
            options.threads = value;
        } else if (argument == "--repeats" || argument == "--iterations") {
            options.repeats = value;
        } else if (argument == "--warmup") {
            options.warmup = value;
        } else if (argument == "--trials") {
            options.trials = value;
        } else if (argument == "--chunk") {
            options.chunk = value;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.nonZerosPerRow > options.cols) {
        throw std::invalid_argument("--nnz-per-row cannot exceed --cols");
    }
    return options;
}

omp_sched_t parseSchedule(const std::string& schedule) {
    if (schedule == "static") {
        return omp_sched_static;
    }
    if (schedule == "dynamic") {
        return omp_sched_dynamic;
    }
    if (schedule == "guided") {
        return omp_sched_guided;
    }
    return omp_sched_auto;
}

std::vector<RowBlock> partitionRows(int totalRows, int ranks) {
    std::vector<RowBlock> blocks(ranks);
    const int baseRows = totalRows / ranks;
    const int remainder = totalRows % ranks;
    int firstRow = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        const int blockRows = baseRows + (rank < remainder ? 1 : 0);
        blocks[rank] = {firstRow, blockRows};
        firstRow += blockRows;
    }
    return blocks;
}

CsrMatrix makeBandedMatrix(int rows, int cols, int nonZerosPerRow) {
    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.rowPtr.resize(rows + 1);
    matrix.columnIndices.reserve(rows * nonZerosPerRow);
    matrix.values.reserve(rows * nonZerosPerRow);

    for (int row = 0; row < rows; ++row) {
        matrix.rowPtr[row] = static_cast<int>(matrix.values.size());
        for (int entry = 0; entry < nonZerosPerRow; ++entry) {
            const int column = (row * 17 + entry) % cols;
            matrix.columnIndices.push_back(column);
            matrix.values.push_back(1.0 / static_cast<double>(1 + ((row + entry) % 13)));
        }
    }
    matrix.rowPtr[rows] = static_cast<int>(matrix.values.size());
    return matrix;
}

CsrMatrix sliceRows(const CsrMatrix& matrix, RowBlock block) {
    CsrMatrix slice;
    slice.rows = block.rows;
    slice.cols = matrix.cols;
    slice.rowPtr.resize(block.rows + 1);
    const int firstEntry = matrix.rowPtr[block.firstRow];
    const int lastEntry = matrix.rowPtr[block.firstRow + block.rows];
    slice.columnIndices.assign(matrix.columnIndices.begin() + firstEntry,
                               matrix.columnIndices.begin() + lastEntry);
    slice.values.assign(matrix.values.begin() + firstEntry, matrix.values.begin() + lastEntry);
    for (int row = 0; row <= block.rows; ++row) {
        slice.rowPtr[row] = matrix.rowPtr[block.firstRow + row] - firstEntry;
    }
    return slice;
}

double denseValue(int row, int column, int epoch = 0) {
    const long long seed = static_cast<long long>(row) * 31 + static_cast<long long>(column) * 7 +
                           static_cast<long long>(epoch) * 11;
    return static_cast<double>((seed % 23) - 11) / 11.0;
}

std::vector<double> makeOwnedDenseRows(RowBlock block, int denseCols, int epoch = 0) {
    std::vector<double> values(static_cast<std::size_t>(block.rows) * denseCols);
    for (int localRow = 0; localRow < block.rows; ++localRow) {
        for (int column = 0; column < denseCols; ++column) {
            values[static_cast<std::size_t>(localRow) * denseCols + column] =
                denseValue(block.firstRow + localRow, column, epoch);
        }
    }
    return values;
}

int ownerOfRow(int row, const std::vector<RowBlock>& blocks) {
    for (int rank = 0; rank < static_cast<int>(blocks.size()); ++rank) {
        if (row >= blocks[rank].firstRow && row < blocks[rank].firstRow + blocks[rank].rows) {
            return rank;
        }
    }
    return -1;
}

void waitAll(std::vector<MPI_Request>& requests, MPI_Comm communicator) {
    if (!requests.empty()) {
        checkMpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE),
                 "MPI_Waitall", communicator);
    }
}

CsrMatrix distributeMatrix(const CsrMatrix* globalMatrix, const std::vector<RowBlock>& blocks, int rank, int ranks,
                           MPI_Comm communicator) {
    if (rank == 0) {
        std::vector<CsrMatrix> slices;
        std::vector<std::array<int, 3>> headers(ranks);
        std::vector<MPI_Request> requests;
        slices.reserve(ranks);
        requests.reserve(4 * (ranks - 1));

        for (int target = 0; target < ranks; ++target) {
            slices.push_back(sliceRows(*globalMatrix, blocks[target]));
            headers[target] = {slices.back().rows, slices.back().cols,
                               static_cast<int>(slices.back().values.size())};
        }
        for (int target = 1; target < ranks; ++target) {
            MPI_Request request = MPI_REQUEST_NULL;
            checkMpi(MPI_Isend(headers[target].data(), 3, MPI_INT, target, kHeaderTag, communicator,
                               &request), "MPI_Isend(matrix header)", communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].rowPtr.data(), headers[target][0] + 1, MPI_INT, target,
                               kRowPtrTag, communicator, &request), "MPI_Isend(row pointers)", communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].columnIndices.data(), headers[target][2], MPI_INT, target,
                               kColumnTag, communicator, &request), "MPI_Isend(column indices)", communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].values.data(), headers[target][2], MPI_DOUBLE, target,
                               kValueTag, communicator, &request), "MPI_Isend(values)", communicator);
            requests.push_back(request);
        }
        waitAll(requests, communicator);
        return std::move(slices.front());
    }

    std::array<int, 3> header{};
    checkMpi(MPI_Recv(header.data(), 3, MPI_INT, 0, kHeaderTag, communicator, MPI_STATUS_IGNORE),
             "MPI_Recv(matrix header)", communicator);
    CsrMatrix localMatrix;
    localMatrix.rows = header[0];
    localMatrix.cols = header[1];
    localMatrix.rowPtr.resize(localMatrix.rows + 1);
    localMatrix.columnIndices.resize(header[2]);
    localMatrix.values.resize(header[2]);
    std::vector<MPI_Request> requests(3, MPI_REQUEST_NULL);
    checkMpi(MPI_Irecv(localMatrix.rowPtr.data(), localMatrix.rows + 1, MPI_INT, 0, kRowPtrTag,
                       communicator, &requests[0]), "MPI_Irecv(row pointers)", communicator);
    checkMpi(MPI_Irecv(localMatrix.columnIndices.data(), header[2], MPI_INT, 0, kColumnTag,
                       communicator, &requests[1]), "MPI_Irecv(column indices)", communicator);
    checkMpi(MPI_Irecv(localMatrix.values.data(), header[2], MPI_DOUBLE, 0, kValueTag,
                       communicator, &requests[2]), "MPI_Irecv(values)", communicator);
    waitAll(requests, communicator);
    return localMatrix;
}

std::unordered_map<int, std::vector<double>> exchangeRemoteDenseRows(
    const CsrMatrix& localMatrix, const std::vector<RowBlock>& blocks, RowBlock localBlock,
    const std::vector<double>& ownedDenseRows, int denseCols, int rank, int ranks,
    MPI_Comm communicator) {
    std::vector<std::vector<int>> requestedRows(ranks);
    for (const int column : localMatrix.columnIndices) {
        const int owner = ownerOfRow(column, blocks);
        if (owner < 0) {
            fail("sparse matrix contains an invalid column index", communicator);
        }
        if (owner != rank) {
            requestedRows[owner].push_back(column);
        }
    }
    for (auto& rows : requestedRows) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }

    std::vector<int> incomingCounts(ranks, 0);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(&incomingCounts[peer], 1, MPI_INT, peer, kRequestCountTag, communicator,
                           &request), "MPI_Irecv(request count)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(requestedRows[peer].data(), 1, MPI_INT, peer, kRequestCountTag, communicator,
                           &request), "MPI_Isend(request count)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<int>> incomingRequests(ranks);
    requests.clear();
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        incomingRequests[peer].resize(incomingCounts[peer]);
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(incomingRequests[peer].data(), incomingCounts[peer], MPI_INT, peer,
                           kRequestRowsTag, communicator, &request), "MPI_Irecv(row requests)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(requestedRows[peer].data(), static_cast<int>(requestedRows[peer].size()),
                           MPI_INT, peer, kRequestRowsTag, communicator, &request),
                 "MPI_Isend(row requests)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<double>> incomingDenseRows(ranks);
    std::vector<std::vector<double>> outgoingDenseRows(ranks);
    requests.clear();
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        incomingDenseRows[peer].resize(static_cast<std::size_t>(requestedRows[peer].size()) * denseCols);
        outgoingDenseRows[peer].resize(static_cast<std::size_t>(incomingRequests[peer].size()) * denseCols);
        for (int index = 0; index < static_cast<int>(incomingRequests[peer].size()); ++index) {
            const int globalRow = incomingRequests[peer][index];
            if (globalRow < localBlock.firstRow || globalRow >= localBlock.firstRow + localBlock.rows) {
                fail("received a request for a non-local dense row", communicator);
            }
            const int localRow = globalRow - localBlock.firstRow;
            std::copy_n(ownedDenseRows.data() + static_cast<std::size_t>(localRow) * denseCols, denseCols,
                        outgoingDenseRows[peer].data() + static_cast<std::size_t>(index) * denseCols);
        }

        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(incomingDenseRows[peer].data(), static_cast<int>(incomingDenseRows[peer].size()),
                           MPI_DOUBLE, peer, kDenseRowsTag, communicator, &request),
                 "MPI_Irecv(dense rows)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(outgoingDenseRows[peer].data(), static_cast<int>(outgoingDenseRows[peer].size()),
                           MPI_DOUBLE, peer, kDenseRowsTag, communicator, &request),
                 "MPI_Isend(dense rows)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    std::unordered_map<int, std::vector<double>> remoteRows;
    for (int peer = 0; peer < ranks; ++peer) {
        for (int index = 0; index < static_cast<int>(requestedRows[peer].size()); ++index) {
            remoteRows.emplace(requestedRows[peer][index],
                               std::vector<double>(incomingDenseRows[peer].begin() +
                                                       static_cast<std::size_t>(index) * denseCols,
                                                   incomingDenseRows[peer].begin() +
                                                       static_cast<std::size_t>(index + 1) * denseCols));
        }
    }
    return remoteRows;
}

void spmm(const CsrMatrix& matrix, RowBlock localBlock, const std::vector<double>& ownedDenseRows,
          const std::unordered_map<int, std::vector<double>>& remoteDenseRows, int denseCols,
          std::vector<double>& result) {
    std::fill(result.begin(), result.end(), 0.0);
#pragma omp parallel for schedule(runtime)
    for (int row = 0; row < matrix.rows; ++row) {
        double* output = result.data() + static_cast<std::size_t>(row) * denseCols;
        for (int entry = matrix.rowPtr[row]; entry < matrix.rowPtr[row + 1]; ++entry) {
            const int denseRow = matrix.columnIndices[entry];
            const double* input = nullptr;
            if (denseRow >= localBlock.firstRow && denseRow < localBlock.firstRow + localBlock.rows) {
                input = ownedDenseRows.data() + static_cast<std::size_t>(denseRow - localBlock.firstRow) * denseCols;
            } else {
                input = remoteDenseRows.at(denseRow).data();
            }
            const double value = matrix.values[entry];
            for (int column = 0; column < denseCols; ++column) {
                output[column] += value * input[column];
            }
        }
    }
}

std::vector<double> gatherResult(const std::vector<double>& localResult, const std::vector<RowBlock>& blocks,
                                 int denseCols, int rank, int ranks,
                                 MPI_Comm communicator) {
    if (rank == 0) {
        std::vector<double> globalResult(static_cast<std::size_t>(blocks.back().firstRow + blocks.back().rows) * denseCols);
        std::copy(localResult.begin(), localResult.end(), globalResult.begin());
        std::vector<MPI_Request> requests;
        requests.reserve(ranks - 1);
        for (int peer = 1; peer < ranks; ++peer) {
            MPI_Request request = MPI_REQUEST_NULL;
            const int count = blocks[peer].rows * denseCols;
            checkMpi(MPI_Irecv(globalResult.data() + static_cast<std::size_t>(blocks[peer].firstRow) * denseCols,
                               count, MPI_DOUBLE, peer, kResultTag, communicator, &request),
                     "MPI_Irecv(result)", communicator);
            requests.push_back(request);
        }
        waitAll(requests, communicator);
        return globalResult;
    }

    MPI_Request request = MPI_REQUEST_NULL;
    checkMpi(MPI_Isend(localResult.data(), static_cast<int>(localResult.size()), MPI_DOUBLE, 0, kResultTag,
                       communicator, &request), "MPI_Isend(result)", communicator);
    checkMpi(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI_Wait(result)", communicator);
    return {};
}

std::vector<double> serialSpmm(const CsrMatrix& matrix, int denseCols, int denseEpoch) {
    std::vector<double> result(static_cast<std::size_t>(matrix.rows) * denseCols, 0.0);
    for (int row = 0; row < matrix.rows; ++row) {
        for (int entry = matrix.rowPtr[row]; entry < matrix.rowPtr[row + 1]; ++entry) {
            for (int column = 0; column < denseCols; ++column) {
                result[static_cast<std::size_t>(row) * denseCols + column] +=
                    matrix.values[entry] * denseValue(matrix.columnIndices[entry], column, denseEpoch);
            }
        }
    }
    return result;
}

double maxAbsoluteDifference(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        maximum = std::max(maximum, std::abs(lhs[index] - rhs[index]));
    }
    return maximum;
}

double maxElapsed(double start, MPI_Comm communicator) {
    const double elapsed = MPI_Wtime() - start;
    double maximum = 0.0;
    checkMpi(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, communicator), "MPI_Reduce",
             communicator);
    return maximum;
}

double percentile90(std::vector<double> samples) {
    if (samples.empty()) {
        throw std::invalid_argument("cannot compute a percentile from no samples");
    }
    std::size_t index = static_cast<std::size_t>(std::ceil(0.90 * samples.size()));
    index = std::max<std::size_t>(1, index) - 1;
    std::nth_element(samples.begin(), samples.begin() + index, samples.end());
    return samples[index];
}

void appendBenchmarkResult(const Options& options, int ranks, const CsrMatrix& matrix,
                           double distributionSeconds, double exchangeSeconds,
                           double communicationP90Seconds, double computeP90Seconds,
                           double endToEndP90Seconds, double gatherSeconds, double gflops,
                           double maxAbsoluteError) {
    const std::filesystem::path outputPath(options.resultsPath);
    const std::filesystem::path parent = outputPath.parent_path();
    std::error_code error;
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            throw std::runtime_error("cannot create results directory: " + parent.string());
        }
    }

    const bool writeHeader = !std::filesystem::exists(outputPath) ||
                             std::filesystem::file_size(outputPath, error) == 0;
    if (error) {
        throw std::runtime_error("cannot inspect results file: " + outputPath.string());
    }

    std::ofstream output(outputPath, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot write results file: " + outputPath.string());
    }
    if (writeHeader) {
        output << "implementation\texperiment\tmatrix_source\trows\tcols\tnnz\tdense_cols\tranks"
               << "\tthreads_per_rank\tomp_schedule\tomp_chunk\twarmup\trepeats\ttrials"
               << "\tdistribution_seconds\thalo_setup_seconds\tcommunication_p90_seconds"
               << "\tcompute_p90_seconds\tend_to_end_p90_seconds\tgather_seconds\tcompute_gflops_p90"
               << "\tmax_abs_error\tvalidation\n";
    }
    output << std::setprecision(17)
           << "mpi_openmp_two_sided\t" << options.experiment << '\t'
           << (options.matrixPath.empty() ? "synthetic" : options.matrixPath) << '\t'
           << matrix.rows << '\t' << matrix.cols << '\t' << matrix.values.size() << '\t'
           << options.denseCols << '\t' << ranks << '\t' << options.threads << '\t'
           << options.schedule << '\t' << options.chunk << '\t' << options.warmup << '\t'
           << options.repeats << '\t' << options.trials << '\t'
           << distributionSeconds << '\t' << exchangeSeconds << '\t' << communicationP90Seconds << '\t'
           << computeP90Seconds << '\t' << endToEndP90Seconds << '\t' << gatherSeconds << '\t'
           << gflops << '\t' << maxAbsoluteError << '\t'
           << (maxAbsoluteError < 1.0e-10 ? "PASS" : "FAIL") << '\n';
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
        const Options options = parseOptions(argc, argv);
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
        checkMpi(MPI_Bcast(matrixShape.data(), static_cast<int>(matrixShape.size()), MPI_INT, 0, communicator),
                 "MPI_Bcast(matrix shape)", communicator);
        const std::vector<RowBlock> outputBlocks = partitionRows(matrixShape[0], ranks);
        const std::vector<RowBlock> denseBlocks = partitionRows(matrixShape[1], ranks);

        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double distributionStart = MPI_Wtime();
        CsrMatrix localMatrix = distributeMatrix(rank == 0 ? &globalMatrix : nullptr, outputBlocks, rank, ranks,
                                                 communicator);
        const double distributionSeconds = maxElapsed(distributionStart, communicator);

        int denseEpoch = 0;
        std::vector<double> ownedDenseRows = makeOwnedDenseRows(denseBlocks[rank], options.denseCols, denseEpoch);
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        const double exchangeStart = MPI_Wtime();
        auto remoteDenseRows = exchangeRemoteDenseRows(localMatrix, denseBlocks, denseBlocks[rank], ownedDenseRows,
                                                        options.denseCols, rank, ranks, communicator);
        const double exchangeSeconds = maxElapsed(exchangeStart, communicator);

        std::vector<double> localResult(static_cast<std::size_t>(localMatrix.rows) * options.denseCols);
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            ++denseEpoch;
            ownedDenseRows = makeOwnedDenseRows(denseBlocks[rank], options.denseCols, denseEpoch);
            remoteDenseRows = exchangeRemoteDenseRows(localMatrix, denseBlocks, denseBlocks[rank], ownedDenseRows,
                                                       options.denseCols, rank, ranks, communicator);
            spmm(localMatrix, denseBlocks[rank], ownedDenseRows, remoteDenseRows, options.denseCols, localResult);
        }

        std::vector<double> computeSamples;
        std::vector<double> communicationSamples;
        std::vector<double> endToEndSamples;
        if (rank == 0) {
            computeSamples.reserve(static_cast<std::size_t>(options.repeats) * options.trials);
            communicationSamples.reserve(static_cast<std::size_t>(options.repeats) * options.trials);
            endToEndSamples.reserve(static_cast<std::size_t>(options.repeats) * options.trials);
        }
        checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
        for (int trial = 0; trial < options.trials; ++trial) {
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                ++denseEpoch;
                ownedDenseRows = makeOwnedDenseRows(denseBlocks[rank], options.denseCols, denseEpoch);
                checkMpi(MPI_Barrier(communicator), "MPI_Barrier", communicator);
                const double start = MPI_Wtime();
                remoteDenseRows = exchangeRemoteDenseRows(localMatrix, denseBlocks, denseBlocks[rank], ownedDenseRows,
                                                           options.denseCols, rank, ranks, communicator);
                const double exchangeEnd = MPI_Wtime();
                spmm(localMatrix, denseBlocks[rank], ownedDenseRows, remoteDenseRows, options.denseCols,
                     localResult);
                const double end = MPI_Wtime();
                const double localTimings[3] = {exchangeEnd - start, end - exchangeEnd, end - start};
                double maxTimings[3] = {};
                checkMpi(MPI_Reduce(localTimings, maxTimings, 3, MPI_DOUBLE, MPI_MAX, 0, communicator),
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
        const std::vector<double> globalResult = gatherResult(localResult, outputBlocks, options.denseCols, rank,
                                                               ranks, communicator);
        const double gatherSeconds = maxElapsed(gatherStart, communicator);

        if (rank == 0) {
            const std::vector<double> reference = serialSpmm(globalMatrix, options.denseCols, denseEpoch);
            const double error = maxAbsoluteDifference(globalResult, reference);
            const double floatingPointOperations = 2.0 * globalMatrix.values.size() * options.denseCols;
            const double computeGflops = floatingPointOperations / computeP90Seconds / 1.0e9;
            appendBenchmarkResult(options, ranks, globalMatrix, distributionSeconds, exchangeSeconds,
                                  communicationP90Seconds, computeP90Seconds, endToEndP90Seconds,
                                  gatherSeconds, computeGflops, error);
            std::cout << std::fixed << std::setprecision(6)
                      << "implementation=mpi_openmp_two_sided\n"
                      << "ranks=" << ranks << " threads_per_rank=" << options.threads
                      << " omp_schedule=" << options.schedule << " omp_chunk=" << options.chunk << '\n'
                      << "experiment=" << options.experiment << " warmup=" << options.warmup
                      << " repeats=" << options.repeats << " trials=" << options.trials << '\n'
                      << "matrix=" << globalMatrix.rows << 'x' << globalMatrix.cols
                      << " nnz=" << globalMatrix.values.size() << " dense_cols=" << options.denseCols << '\n'
                      << "matrix_source=" << (options.matrixPath.empty() ? "synthetic" : options.matrixPath) << '\n'
                      << "distribution_seconds=" << distributionSeconds << '\n'
                      << "halo_setup_seconds=" << exchangeSeconds << '\n'
                      << "communication_p90_seconds=" << communicationP90Seconds << '\n'
                      << "compute_p90_seconds=" << computeP90Seconds << '\n'
                      << "end_to_end_p90_seconds=" << endToEndP90Seconds << '\n'
                      << "gather_seconds=" << gatherSeconds << '\n'
                      << "compute_gflops=" << computeGflops << '\n'
                      << "max_abs_error=" << error << '\n'
                      << "validation=" << (error < 1.0e-10 ? "PASS" : "FAIL") << '\n'
                      << "results_file=" << options.resultsPath << '\n';
        }
    } catch (const std::exception& error) {
        fail(error.what(), communicator);
    }

    checkMpi(MPI_Finalize(), "MPI_Finalize", communicator);
    return EXIT_SUCCESS;
}
