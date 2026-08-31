#include "spmm_common.hpp"

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
#include <utility>
#include <vector>

namespace {

constexpr int kHeaderTag = 100;
constexpr int kRowPtrTag = 101;
constexpr int kColumnTag = 102;
constexpr int kValueTag = 103;
constexpr int kResultTag = 300;

void printUsage(const std::string& programName, const std::string& defaultResultsPath) {
    std::cout
        << "Usage: " << programName << " [options]\n"
        << "  --rows N          Rows of sparse matrix A (default: 1024)\n"
        << "  --cols N          Columns of sparse matrix A / rows of B (default: 1024)\n"
        << "  --dense-cols N    Columns of dense matrix B (default: 32)\n"
        << "  --nnz-per-row N   Non-zeros in each row of A (default: 16)\n"
        << "  --threads N       OpenMP threads per MPI rank (default: 1)\n"
        << "  --warmup N        Untimed SpMM repetitions before measurement (default: 2)\n"
        << "  --repeats N       Timed repetitions in each trial (default: 10)\n"
        << "  --trials N        Number of independent trials (default: 5)\n"
        << "  --matrix PATH     Matrix Market coordinate input; synthetic matrix if omitted\n"
        << "  --results PATH    TSV output path (default: " << defaultResultsPath << ")\n"
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

template <typename T>
T* dataOrNull(std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

template <typename T>
const T* dataOrNull(const std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

}  // namespace

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

Options parseOptions(int argc, char** argv, const std::string& programName,
                     const std::string& defaultResultsPath) {
    Options options;
    options.resultsPath = defaultResultsPath;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            printUsage(programName, defaultResultsPath);
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

int ownerOfRow(int row, const std::vector<RowBlock>& blocks) {
    for (int rank = 0; rank < static_cast<int>(blocks.size()); ++rank) {
        if (row >= blocks[rank].firstRow && row < blocks[rank].firstRow + blocks[rank].rows) {
            return rank;
        }
    }
    return -1;
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

CsrMatrix distributeMatrix(const CsrMatrix* globalMatrix, const std::vector<RowBlock>& blocks,
                           int rank, int ranks, MPI_Comm communicator) {
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
                               kRowPtrTag, communicator, &request), "MPI_Isend(row pointers)",
                     communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].columnIndices.data(), headers[target][2], MPI_INT,
                               target, kColumnTag, communicator, &request),
                     "MPI_Isend(column indices)", communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].values.data(), headers[target][2], MPI_DOUBLE, target,
                               kValueTag, communicator, &request), "MPI_Isend(values)",
                     communicator);
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
    checkMpi(MPI_Irecv(dataOrNull(localMatrix.rowPtr), localMatrix.rows + 1, MPI_INT, 0, kRowPtrTag,
                       communicator, &requests[0]), "MPI_Irecv(row pointers)", communicator);
    checkMpi(MPI_Irecv(dataOrNull(localMatrix.columnIndices), header[2], MPI_INT, 0, kColumnTag,
                       communicator, &requests[1]), "MPI_Irecv(column indices)", communicator);
    checkMpi(MPI_Irecv(dataOrNull(localMatrix.values), header[2], MPI_DOUBLE, 0, kValueTag,
                       communicator, &requests[2]), "MPI_Irecv(values)", communicator);
    waitAll(requests, communicator);
    return localMatrix;
}

double denseValue(int row, int column, int epoch) {
    const long long seed = static_cast<long long>(row) * 31 + static_cast<long long>(column) * 7 +
                           static_cast<long long>(epoch) * 11;
    return static_cast<double>((seed % 23) - 11) / 11.0;
}

void fillOwnedDenseRows(RowBlock block, int denseCols, int epoch, std::vector<double>& values) {
    values.resize(static_cast<std::size_t>(block.rows) * denseCols);
    for (int localRow = 0; localRow < block.rows; ++localRow) {
        for (int column = 0; column < denseCols; ++column) {
            values[static_cast<std::size_t>(localRow) * denseCols + column] =
                denseValue(block.firstRow + localRow, column, epoch);
        }
    }
}

std::vector<double> makeOwnedDenseRows(RowBlock block, int denseCols, int epoch) {
    std::vector<double> values;
    fillOwnedDenseRows(block, denseCols, epoch, values);
    return values;
}

void waitAll(std::vector<MPI_Request>& requests, MPI_Comm communicator) {
    if (!requests.empty()) {
        checkMpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE),
                 "MPI_Waitall", communicator);
    }
}

std::vector<double> gatherResult(const std::vector<double>& localResult,
                                 const std::vector<RowBlock>& blocks, int denseCols, int rank,
                                 int ranks, MPI_Comm communicator) {
    if (rank == 0) {
        std::vector<double> globalResult(
            static_cast<std::size_t>(blocks.back().firstRow + blocks.back().rows) * denseCols);
        std::copy(localResult.begin(), localResult.end(), globalResult.begin());
        std::vector<MPI_Request> requests;
        requests.reserve(ranks - 1);
        for (int peer = 1; peer < ranks; ++peer) {
            MPI_Request request = MPI_REQUEST_NULL;
            const int count = blocks[peer].rows * denseCols;
            checkMpi(MPI_Irecv(globalResult.data() +
                                   static_cast<std::size_t>(blocks[peer].firstRow) * denseCols,
                               count, MPI_DOUBLE, peer, kResultTag, communicator, &request),
                     "MPI_Irecv(result)", communicator);
            requests.push_back(request);
        }
        waitAll(requests, communicator);
        return globalResult;
    }

    MPI_Request request = MPI_REQUEST_NULL;
    checkMpi(MPI_Isend(dataOrNull(localResult), static_cast<int>(localResult.size()), MPI_DOUBLE, 0,
                       kResultTag, communicator, &request),
             "MPI_Isend(result)", communicator);
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

void appendBenchmarkResult(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrix, double distributionSeconds,
                           double haloSetupSeconds, double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double gflops, double maxAbsoluteError) {
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
           << implementation << '\t' << options.experiment << '\t'
           << (options.matrixPath.empty() ? "synthetic" : options.matrixPath) << '\t'
           << matrix.rows << '\t' << matrix.cols << '\t' << matrix.values.size() << '\t'
           << options.denseCols << '\t' << ranks << '\t' << options.threads << '\t'
           << options.schedule << '\t' << options.chunk << '\t' << options.warmup << '\t'
           << options.repeats << '\t' << options.trials << '\t'
           << distributionSeconds << '\t' << haloSetupSeconds << '\t'
           << communicationP90Seconds << '\t' << computeP90Seconds << '\t'
           << endToEndP90Seconds << '\t' << gatherSeconds << '\t' << gflops << '\t'
           << maxAbsoluteError << '\t' << (maxAbsoluteError < 1.0e-10 ? "PASS" : "FAIL") << '\n';
}

void printBenchmarkSummary(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrix, double distributionSeconds,
                           double haloSetupSeconds, double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double computeGflops, double maxAbsoluteError) {
    std::cout << std::fixed << std::setprecision(6)
              << "implementation=" << implementation << '\n'
              << "ranks=" << ranks << " threads_per_rank=" << options.threads
              << " omp_schedule=" << options.schedule << " omp_chunk=" << options.chunk << '\n'
              << "experiment=" << options.experiment << " warmup=" << options.warmup
              << " repeats=" << options.repeats << " trials=" << options.trials << '\n'
              << "matrix=" << matrix.rows << 'x' << matrix.cols << " nnz=" << matrix.values.size()
              << " dense_cols=" << options.denseCols << '\n'
              << "matrix_source=" << (options.matrixPath.empty() ? "synthetic" : options.matrixPath)
              << '\n'
              << "distribution_seconds=" << distributionSeconds << '\n'
              << "halo_setup_seconds=" << haloSetupSeconds << '\n'
              << "communication_p90_seconds=" << communicationP90Seconds << '\n'
              << "compute_p90_seconds=" << computeP90Seconds << '\n'
              << "end_to_end_p90_seconds=" << endToEndP90Seconds << '\n'
              << "gather_seconds=" << gatherSeconds << '\n'
              << "compute_gflops=" << computeGflops << '\n'
              << "max_abs_error=" << maxAbsoluteError << '\n'
              << "validation=" << (maxAbsoluteError < 1.0e-10 ? "PASS" : "FAIL") << '\n'
              << "results_file=" << options.resultsPath << '\n';
}
