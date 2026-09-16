#include "spgemm_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
constexpr int kResultHeaderTag = 300;
constexpr int kResultRowPtrTag = 301;
constexpr int kResultColumnTag = 302;
constexpr int kResultValueTag = 303;

std::string matrixBSource(const Options& options) {
    if (!options.matrixBPath.empty()) {
        return options.matrixBPath;
    }
    if (!options.matrixAPath.empty()) {
        return options.matrixAPath;
    }
    return "synthetic";
}

const char* validationStatus(std::optional<double> maxAbsoluteError) {
    if (!maxAbsoluteError) {
        return "SKIPPED";
    }
    return validationPassed(*maxAbsoluteError) ? "PASS" : "FAIL";
}

void printUsage(const std::string& programName, const std::string& defaultResultsPath) {
    std::cout
        << "Usage: " << programName << " [options]\n"
        << "  --rows N             Rows of synthetic sparse matrix A (default: 1024)\n"
        << "  --cols N             Columns of A / rows of B (default: 1024)\n"
        << "  --b-cols N           Columns of synthetic sparse matrix B (default: 1024)\n"
        << "  --nnz-per-row N      Non-zeros in each synthetic row of A (default: 16)\n"
        << "  --b-nnz-per-row N    Non-zeros in each synthetic row of B (default: 16)\n"
        << "  --threads N          OpenMP threads per MPI rank (default: 1)\n"
        << "  --warmup N           Extra untimed repetitions after the first product (default: 2)\n"
        << "  --repeats N          Timed repetitions in each trial (default: 10)\n"
        << "  --trials N           Groups of repetitions within the same MPI run (default: 5)\n"
        << "  --matrix PATH        Matrix Market input used for both A and B; must be square\n"
        << "  --matrix-a PATH      Matrix Market coordinate input for A\n"
        << "                       Without --matrix-b, requires square A and computes A*A\n"
        << "  --matrix-b PATH      Matrix Market coordinate input for B\n"
        << "  --no-validate        Skip serial reference and result comparison (default: validate)\n"
        << "  --results PATH       TSV output path (default: " << defaultResultsPath << ")\n"
        << "  --experiment TAG     Experiment label stored in the TSV (default: manual)\n"
        << "  --schedule NAME      OpenMP schedule: static, dynamic, guided, auto (default: guided)\n"
        << "  --chunk N            OpenMP schedule chunk, ignored by auto (default: 64)\n";
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

void validateMultiplicationDimensions(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    if (matrixA.cols != matrixB.rows) {
        throw std::invalid_argument("SpGEMM requires A.cols == B.rows");
    }
}

void appendCsrBlock(CsrMatrix& globalResult, const CsrMatrix& localResult, RowBlock block,
                    int resultCols) {
    if (localResult.rows != block.rows || localResult.cols != resultCols) {
        throw std::runtime_error("received a result block with an unexpected shape");
    }

    for (int row = 0; row < localResult.rows; ++row) {
        const int first = localResult.rowPtr[row];
        const int last = localResult.rowPtr[row + 1];
        globalResult.rowPtr[block.firstRow + row + 1] =
            globalResult.rowPtr[block.firstRow + row] + (last - first);
        globalResult.columnIndices.insert(globalResult.columnIndices.end(),
                                          localResult.columnIndices.begin() + first,
                                          localResult.columnIndices.begin() + last);
        globalResult.values.insert(globalResult.values.end(), localResult.values.begin() + first,
                                   localResult.values.begin() + last);
    }
}

CsrMatrix buildSerialRowWiseProduct(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    validateMultiplicationDimensions(matrixA, matrixB);

    CsrMatrix result;
    result.rows = matrixA.rows;
    result.cols = matrixB.cols;
    result.rowPtr.resize(matrixA.rows + 1, 0);

    std::unordered_map<int, double> accumulator;
    std::vector<int> touchedColumns;
    for (int row = 0; row < matrixA.rows; ++row) {
        accumulator.clear();
        touchedColumns.clear();

        for (int aEntry = matrixA.rowPtr[row]; aEntry < matrixA.rowPtr[row + 1]; ++aEntry) {
            const int bRow = matrixA.columnIndices[aEntry];
            if (bRow < 0 || bRow >= matrixB.rows) {
                throw std::runtime_error("A contains a column index outside B's row range");
            }
            const double aValue = matrixA.values[aEntry];
            for (int bEntry = matrixB.rowPtr[bRow]; bEntry < matrixB.rowPtr[bRow + 1]; ++bEntry) {
                const int column = matrixB.columnIndices[bEntry];
                const double product = aValue * matrixB.values[bEntry];
                const auto [position, inserted] = accumulator.emplace(column, product);
                if (inserted) {
                    touchedColumns.push_back(column);
                } else {
                    position->second += product;
                }
            }
        }

        std::sort(touchedColumns.begin(), touchedColumns.end());
        for (const int column : touchedColumns) {
            const double value = accumulator.at(column);
            if (value != 0.0) {
                result.columnIndices.push_back(column);
                result.values.push_back(value);
            }
        }
        result.rowPtr[row + 1] = static_cast<int>(result.values.size());
    }
    return result;
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
        if (argument == "--no-validate") {
            options.validate = false;
            continue;
        }
        if (index + 1 == argc) {
            throw std::invalid_argument("missing value for " + argument);
        }

        if (argument == "--schedule" || argument == "--matrix" || argument == "--matrix-a" ||
            argument == "--matrix-b" || argument == "--results" || argument == "--experiment") {
            const std::string value = argv[++index];
            if (argument == "--matrix") {
                options.matrixAPath = value;
                options.matrixBPath = value;
                continue;
            }
            if (argument == "--matrix-a") {
                options.matrixAPath = value;
                continue;
            }
            if (argument == "--matrix-b") {
                options.matrixBPath = value;
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
        } else if (argument == "--b-cols") {
            options.bCols = value;
        } else if (argument == "--nnz-per-row") {
            options.nonZerosPerRow = value;
        } else if (argument == "--b-nnz-per-row") {
            options.bNonZerosPerRow = value;
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
    if (options.bNonZerosPerRow > options.bCols) {
        throw std::invalid_argument("--b-nnz-per-row cannot exceed --b-cols");
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
    matrix.columnIndices.reserve(static_cast<std::size_t>(rows) * nonZerosPerRow);
    matrix.values.reserve(static_cast<std::size_t>(rows) * nonZerosPerRow);

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
                               &request),
                     "MPI_Isend(matrix header)", communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].rowPtr.data(), headers[target][0] + 1, MPI_INT,
                               target, kRowPtrTag, communicator, &request),
                     "MPI_Isend(row pointers)", communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].columnIndices.data(), headers[target][2], MPI_INT,
                               target, kColumnTag, communicator, &request),
                     "MPI_Isend(column indices)", communicator);
            requests.push_back(request);
            checkMpi(MPI_Isend(slices[target].values.data(), headers[target][2], MPI_DOUBLE, target,
                               kValueTag, communicator, &request),
                     "MPI_Isend(values)", communicator);
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
    checkMpi(MPI_Irecv(dataOrNull(localMatrix.rowPtr), localMatrix.rows + 1, MPI_INT, 0,
                       kRowPtrTag, communicator, &requests[0]),
             "MPI_Irecv(row pointers)", communicator);
    checkMpi(MPI_Irecv(dataOrNull(localMatrix.columnIndices), header[2], MPI_INT, 0, kColumnTag,
                       communicator, &requests[1]),
             "MPI_Irecv(column indices)", communicator);
    checkMpi(MPI_Irecv(dataOrNull(localMatrix.values), header[2], MPI_DOUBLE, 0, kValueTag,
                       communicator, &requests[2]),
             "MPI_Irecv(values)", communicator);
    waitAll(requests, communicator);
    return localMatrix;
}

void waitAll(std::vector<MPI_Request>& requests, MPI_Comm communicator) {
    if (!requests.empty()) {
        checkMpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE),
                 "MPI_Waitall", communicator);
    }
}

CsrMatrix gatherCsrMatrix(const CsrMatrix& localResult, const std::vector<RowBlock>& blocks,
                          int resultCols, int rank, int ranks, MPI_Comm communicator) {
    if (rank == 0) {
        CsrMatrix globalResult;
        globalResult.rows = blocks.back().firstRow + blocks.back().rows;
        globalResult.cols = resultCols;
        globalResult.rowPtr.assign(globalResult.rows + 1, 0);

        appendCsrBlock(globalResult, localResult, blocks[0], resultCols);

        for (int peer = 1; peer < ranks; ++peer) {
            std::array<int, 3> header{};
            checkMpi(MPI_Recv(header.data(), 3, MPI_INT, peer, kResultHeaderTag, communicator,
                              MPI_STATUS_IGNORE),
                     "MPI_Recv(result header)", communicator);

            CsrMatrix peerResult;
            peerResult.rows = header[0];
            peerResult.cols = header[1];
            peerResult.rowPtr.resize(peerResult.rows + 1);
            peerResult.columnIndices.resize(header[2]);
            peerResult.values.resize(header[2]);

            checkMpi(MPI_Recv(dataOrNull(peerResult.rowPtr), peerResult.rows + 1, MPI_INT, peer,
                              kResultRowPtrTag, communicator, MPI_STATUS_IGNORE),
                     "MPI_Recv(result row pointers)", communicator);
            checkMpi(MPI_Recv(dataOrNull(peerResult.columnIndices), header[2], MPI_INT, peer,
                              kResultColumnTag, communicator, MPI_STATUS_IGNORE),
                     "MPI_Recv(result column indices)", communicator);
            checkMpi(MPI_Recv(dataOrNull(peerResult.values), header[2], MPI_DOUBLE, peer,
                              kResultValueTag, communicator, MPI_STATUS_IGNORE),
                     "MPI_Recv(result values)", communicator);
            appendCsrBlock(globalResult, peerResult, blocks[peer], resultCols);
        }
        return globalResult;
    }

    const std::array<int, 3> header = {localResult.rows, localResult.cols,
                                       static_cast<int>(localResult.values.size())};
    checkMpi(MPI_Send(header.data(), 3, MPI_INT, 0, kResultHeaderTag, communicator),
             "MPI_Send(result header)", communicator);
    checkMpi(MPI_Send(dataOrNull(localResult.rowPtr), localResult.rows + 1, MPI_INT, 0,
                      kResultRowPtrTag, communicator),
             "MPI_Send(result row pointers)", communicator);
    checkMpi(MPI_Send(dataOrNull(localResult.columnIndices), header[2], MPI_INT, 0,
                      kResultColumnTag, communicator),
             "MPI_Send(result column indices)", communicator);
    checkMpi(MPI_Send(dataOrNull(localResult.values), header[2], MPI_DOUBLE, 0, kResultValueTag,
                      communicator),
             "MPI_Send(result values)", communicator);
    return {};
}

CsrMatrix serialSpgemm(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    return buildSerialRowWiseProduct(matrixA, matrixB);
}

std::int64_t scalarMultiplicationCount(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    validateMultiplicationDimensions(matrixA, matrixB);
    std::int64_t count = 0;
    for (int row = 0; row < matrixA.rows; ++row) {
        for (int entry = matrixA.rowPtr[row]; entry < matrixA.rowPtr[row + 1]; ++entry) {
            const int bRow = matrixA.columnIndices[entry];
            if (bRow < 0 || bRow >= matrixB.rows) {
                throw std::runtime_error("A contains a column index outside B's row range");
            }
            count += matrixB.rowPtr[bRow + 1] - matrixB.rowPtr[bRow];
        }
    }
    return count;
}

double maxAbsoluteDifference(const CsrMatrix& lhs, const CsrMatrix& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return std::numeric_limits<double>::infinity();
    }

    const auto isFinite = [](double value) { return std::isfinite(value); };
    if (!std::all_of(lhs.values.begin(), lhs.values.end(), isFinite) ||
        !std::all_of(rhs.values.begin(), rhs.values.end(), isFinite)) {
        return std::numeric_limits<double>::infinity();
    }

    double maximum = 0.0;
    for (int row = 0; row < lhs.rows; ++row) {
        int left = lhs.rowPtr[row];
        int right = rhs.rowPtr[row];
        const int leftEnd = lhs.rowPtr[row + 1];
        const int rightEnd = rhs.rowPtr[row + 1];

        while (left < leftEnd || right < rightEnd) {
            double difference = 0.0;
            if (right == rightEnd ||
                (left < leftEnd && lhs.columnIndices[left] < rhs.columnIndices[right])) {
                difference = std::abs(lhs.values[left]);
                ++left;
            } else if (left == leftEnd || rhs.columnIndices[right] < lhs.columnIndices[left]) {
                difference = std::abs(rhs.values[right]);
                ++right;
            } else {
                difference = std::abs(lhs.values[left] - rhs.values[right]);
                ++left;
                ++right;
            }
            if (!std::isfinite(difference)) {
                return std::numeric_limits<double>::infinity();
            }
            maximum = std::max(maximum, difference);
        }
    }
    return maximum;
}

bool validationPassed(double maxAbsoluteError) {
    return std::isfinite(maxAbsoluteError) && maxAbsoluteError >= 0.0 &&
           maxAbsoluteError < 1.0e-10;
}

double maxRankValue(double value, MPI_Comm communicator) {
    double maximum = 0.0;
    checkMpi(MPI_Reduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, communicator), "MPI_Reduce",
             communicator);
    return maximum;
}

double maxElapsed(double start, MPI_Comm communicator) {
    return maxRankValue(MPI_Wtime() - start, communicator);
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
                           const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                           const CsrMatrix& matrixC, double distributionSeconds,
                           double haloSetupSeconds, double firstProductSeconds,
                           double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double gflops,
                           std::optional<double> maxAbsoluteError) {
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

    const std::string header =
        "implementation\texperiment\tmatrix_a_source\tmatrix_b_source"
        "\ta_rows\ta_cols\tb_rows\tb_cols\ta_nnz\tb_nnz\tc_nnz\tranks"
        "\tthreads_per_rank\tomp_schedule\tomp_chunk\twarmup\trepeats\ttrials"
        "\tdistribution_seconds\thalo_setup_seconds\tfirst_product_seconds"
        "\tcommunication_p90_seconds\tcompute_p90_seconds\tend_to_end_p90_seconds"
        "\tgather_seconds\tcompute_gflops_p90\tmax_abs_error\tvalidation\tbenchmark_protocol";
    if (!writeHeader) {
        std::ifstream existing(outputPath);
        std::string existingHeader;
        std::getline(existing, existingHeader);
        if (!existingHeader.empty() && existingHeader.back() == '\r') {
            existingHeader.pop_back();
        }
        if (existingHeader != header) {
            throw std::runtime_error("incompatible benchmark TSV header; choose a new --results path: " +
                                     outputPath.string());
        }
    }

    std::ofstream output(outputPath, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot write results file: " + outputPath.string());
    }
    if (writeHeader) {
        output << header << '\n';
    }
    output << std::setprecision(17) << implementation << '\t' << options.experiment << '\t'
           << (options.matrixAPath.empty() ? "synthetic" : options.matrixAPath) << '\t'
           << matrixBSource(options) << '\t'
           << matrixA.rows << '\t' << matrixA.cols << '\t' << matrixB.rows << '\t'
           << matrixB.cols << '\t' << matrixA.values.size() << '\t' << matrixB.values.size()
           << '\t' << matrixC.values.size() << '\t' << ranks << '\t' << options.threads << '\t'
           << options.schedule << '\t' << options.chunk << '\t' << options.warmup << '\t'
           << options.repeats << '\t' << options.trials << '\t' << distributionSeconds << '\t'
           << haloSetupSeconds << '\t' << firstProductSeconds << '\t'
           << communicationP90Seconds << '\t' << computeP90Seconds
           << '\t' << endToEndP90Seconds << '\t' << gatherSeconds << '\t' << gflops << '\t';
    if (maxAbsoluteError) {
        output << *maxAbsoluteError;
    } else {
        output << "NA";
    }
    output << '\t' << validationStatus(maxAbsoluteError) << "\tprepared_halo_v2\n";
}

void printBenchmarkSummary(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                           const CsrMatrix& matrixC, double distributionSeconds,
                           double haloSetupSeconds, double firstProductSeconds,
                           double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double computeGflops,
                           std::optional<double> maxAbsoluteError) {
    std::cout << std::fixed << std::setprecision(6) << "implementation=" << implementation << '\n'
              << "benchmark_protocol=prepared_halo_v2\n"
              << "ranks=" << ranks << " threads_per_rank=" << options.threads
              << " omp_schedule=" << options.schedule << " omp_chunk=" << options.chunk << '\n'
              << "experiment=" << options.experiment << " warmup=" << options.warmup
              << " repeats=" << options.repeats << " trials=" << options.trials << '\n'
              << "A=" << matrixA.rows << 'x' << matrixA.cols << " nnz=" << matrixA.values.size()
              << '\n'
              << "B=" << matrixB.rows << 'x' << matrixB.cols << " nnz=" << matrixB.values.size()
              << '\n'
              << "C=" << matrixC.rows << 'x' << matrixC.cols << " nnz=" << matrixC.values.size()
              << '\n'
              << "matrix_a_source="
              << (options.matrixAPath.empty() ? "synthetic" : options.matrixAPath) << '\n'
              << "matrix_b_source="
              << matrixBSource(options) << '\n'
              << "distribution_seconds=" << distributionSeconds << '\n'
              << "halo_setup_seconds=" << haloSetupSeconds << '\n'
              << "first_product_seconds=" << firstProductSeconds << '\n'
              << "communication_p90_seconds=" << communicationP90Seconds << '\n'
              << "compute_p90_seconds=" << computeP90Seconds << '\n'
              << "end_to_end_p90_seconds=" << endToEndP90Seconds << '\n'
              << "gather_seconds=" << gatherSeconds << '\n'
              << "compute_gflops=" << computeGflops << '\n'
              << "max_abs_error=";
    if (maxAbsoluteError) {
        std::cout << *maxAbsoluteError;
    } else {
        std::cout << "NA";
    }
    std::cout << '\n'
              << "validation=" << validationStatus(maxAbsoluteError) << '\n'
              << "results_file=" << options.resultsPath << '\n';
}
