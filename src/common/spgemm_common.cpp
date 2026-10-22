#include "spgemm_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
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
        << "  --partial            Compare only row nnz counts with a streamed serial reference\n"
        << "  --partial-cache DIR  Require a prepared partial reference (only with --partial)\n"
        << "  --no-validate        Skip serial reference and result comparison (default: full)\n"
        << "  --results PATH       TSV output path (default: " << defaultResultsPath << ")\n"
        << "  --experiment TAG     Experiment label stored in the TSV (default: manual)\n"
        << "  --schedule NAME      OpenMP schedule: static, dynamic, guided, auto (default: guided)\n"
        << "  --chunk N            OpenMP schedule chunk, ignored by auto (default: 64)\n";
}

int parsePositiveInt(const char* value, const char* option) {
    try {
        const std::string input(value);
        std::size_t used = 0;
        const int parsed = std::stoi(input, &used);
        if (used != input.size() || parsed <= 0) {
            throw std::invalid_argument("not a positive integer");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects a positive integer");
    }
}

int parseNonNegativeInt(const char* value, const char* option) {
    try {
        const std::string input(value);
        std::size_t used = 0;
        const int parsed = std::stoi(input, &used);
        if (used != input.size() || parsed < 0) {
            throw std::invalid_argument("not a non-negative integer");
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
        const CsrOffset first = localResult.rowPtr[row];
        const CsrOffset last = localResult.rowPtr[row + 1];
        globalResult.rowPtr[block.firstRow + row + 1] = checkedCsrAdd(
            globalResult.rowPtr[block.firstRow + row], last - first, "global result prefix");
        globalResult.columnIndices.insert(globalResult.columnIndices.end(),
                                          localResult.columnIndices.begin() + first,
                                          localResult.columnIndices.begin() + last);
        globalResult.values.insert(globalResult.values.end(), localResult.values.begin() + first,
                                   localResult.values.begin() + last);
    }
}

template <typename ConsumeRow>
void forEachSerialProductRow(const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                             ConsumeRow consumeRow) {
    validateMultiplicationDimensions(matrixA, matrixB);
    std::unordered_map<int, double> accumulator;
    for (int row = 0; row < matrixA.rows; ++row) {
        accumulator.clear();

        for (CsrOffset aEntry = matrixA.rowPtr[row]; aEntry < matrixA.rowPtr[row + 1]; ++aEntry) {
            const int bRow = matrixA.columnIndices[aEntry];
            if (bRow < 0 || bRow >= matrixB.rows) {
                throw std::runtime_error("A contains a column index outside B's row range");
            }
            const double aValue = matrixA.values[aEntry];
            for (CsrOffset bEntry = matrixB.rowPtr[bRow]; bEntry < matrixB.rowPtr[bRow + 1]; ++bEntry) {
                const int column = matrixB.columnIndices[bEntry];
                const double product = aValue * matrixB.values[bEntry];
                const auto [position, inserted] = accumulator.emplace(column, product);
                if (!inserted) {
                    position->second += product;
                }
            }
        }
        consumeRow(row, accumulator);
    }
}

CsrMatrix buildSerialRowWiseProduct(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    validateMultiplicationDimensions(matrixA, matrixB);
    CsrMatrix result;
    result.rows = matrixA.rows;
    result.cols = matrixB.cols;
    result.rowPtr.resize(static_cast<std::size_t>(matrixA.rows) + 1, 0);
    std::vector<int> touchedColumns;
    forEachSerialProductRow(matrixA, matrixB, [&](int row, const auto& accumulator) {
        touchedColumns.clear();
        for (const auto& entry : accumulator) {
            if (entry.second != 0.0) touchedColumns.push_back(entry.first);
        }
        std::sort(touchedColumns.begin(), touchedColumns.end());
        for (const int column : touchedColumns) {
            result.columnIndices.push_back(column);
            result.values.push_back(accumulator.at(column));
        }
        result.rowPtr[row + 1] = checkedCsrCount(result.values.size());
    });
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
        if (argument == "--no-validate" || argument == "--partial") {
            const auto mode = argument == "--partial" ? ValidationMode::Partial : ValidationMode::None;
            if (options.validationMode != ValidationMode::Full && options.validationMode != mode) {
                throw std::invalid_argument("--partial and --no-validate are mutually exclusive");
            }
            options.validationMode = mode;
            continue;
        }
        if (index + 1 == argc) {
            throw std::invalid_argument("missing value for " + argument);
        }

        if (argument == "--schedule" || argument == "--matrix" || argument == "--matrix-a" ||
            argument == "--matrix-b" || argument == "--results" || argument == "--experiment" ||
            argument == "--partial-cache") {
            const std::string value = argv[++index];
            if (argument == "--partial-cache") {
                if (value.empty()) throw std::invalid_argument("--partial-cache requires a directory");
                options.partialCacheDirectory = value;
                continue;
            }
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
    if (!options.partialCacheDirectory.empty() && options.validationMode != ValidationMode::Partial)
        throw std::invalid_argument("--partial-cache requires --partial");
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
    const auto count = checkedCsrCount(static_cast<std::size_t>(rows) * nonZerosPerRow);
    matrix.rowPtr.resize(static_cast<std::size_t>(rows) + 1);
    matrix.columnIndices.reserve(csrSize<int>(count));
    matrix.values.reserve(csrSize<double>(count));

    for (int row = 0; row < rows; ++row) {
        matrix.rowPtr[row] = checkedCsrCount(matrix.values.size());
        for (int entry = 0; entry < nonZerosPerRow; ++entry) {
            const int column = static_cast<int>((static_cast<std::int64_t>(row) * 17 + entry) % cols);
            matrix.columnIndices.push_back(column);
            matrix.values.push_back(1.0 / static_cast<double>(1 + ((static_cast<std::int64_t>(row) + entry) % 13)));
        }
    }
    matrix.rowPtr[rows] = checkedCsrCount(matrix.values.size());
    return matrix;
}

CsrMatrix sliceRows(const CsrMatrix& matrix, RowBlock block) {
    CsrMatrix slice;
    slice.rows = block.rows;
    slice.cols = matrix.cols;
    slice.rowPtr.resize(static_cast<std::size_t>(block.rows) + 1);
    const CsrOffset firstEntry = matrix.rowPtr[block.firstRow];
    const CsrOffset lastEntry = matrix.rowPtr[block.firstRow + block.rows];
    slice.columnIndices.assign(matrix.columnIndices.begin() + firstEntry,
                               matrix.columnIndices.begin() + lastEntry);
    slice.values.assign(matrix.values.begin() + firstEntry, matrix.values.begin() + lastEntry);
    for (std::size_t row = 0; row <= static_cast<std::size_t>(block.rows); ++row) {
        slice.rowPtr[row] = matrix.rowPtr[block.firstRow + row] - firstEntry;
    }
    return slice;
}

CsrMatrix distributeMatrix(const CsrMatrix* globalMatrix, const std::vector<RowBlock>& blocks,
                           int rank, int ranks, MPI_Comm communicator) {
    if (rank == 0) {
        std::vector<CsrMatrix> slices;
        std::vector<std::array<CsrOffset, 3>> headers(ranks);
        std::vector<MPI_Request> requests;
        slices.reserve(ranks);
        requests.reserve(4 * (ranks - 1));

        for (int target = 0; target < ranks; ++target) {
            slices.push_back(sliceRows(*globalMatrix, blocks[target]));
            headers[target] = {slices.back().rows, slices.back().cols,
                               checkedCsrCount(slices.back().values.size())};
        }
        for (int target = 1; target < ranks; ++target) {
            csrIsend(headers[target].data(), 3, target, kHeaderTag, communicator, requests);
            csrIsend(slices[target].rowPtr.data(), checkedCsrCount(slices[target].rowPtr.size()), target, kRowPtrTag, communicator, requests);
            csrIsend(slices[target].columnIndices.data(), headers[target][2], target, kColumnTag, communicator, requests);
            csrIsend(slices[target].values.data(), headers[target][2], target, kValueTag, communicator, requests);
        }
        waitAll(requests, communicator);
        return std::move(slices.front());
    }

    std::array<CsrOffset, 3> header{};
    csrRecv(header.data(), 3, 0, kHeaderTag, communicator);
    CsrMatrix localMatrix;
    localMatrix.rows = checkedMatrixDimension(header[0]);
    localMatrix.cols = checkedMatrixDimension(header[1]);
    localMatrix.rowPtr.resize(static_cast<std::size_t>(localMatrix.rows) + 1);
    localMatrix.columnIndices.resize(csrSize<int>(header[2]));
    localMatrix.values.resize(csrSize<double>(header[2]));
    std::vector<MPI_Request> requests;
    csrIrecv(dataOrNull(localMatrix.rowPtr), checkedCsrCount(localMatrix.rowPtr.size()), 0, kRowPtrTag, communicator, requests);
    csrIrecv(dataOrNull(localMatrix.columnIndices), header[2], 0, kColumnTag, communicator, requests);
    csrIrecv(dataOrNull(localMatrix.values), header[2], 0, kValueTag, communicator, requests);
    waitAll(requests, communicator);
    return localMatrix;
}

void waitAll(std::vector<MPI_Request>& requests, MPI_Comm communicator) {
    forMpiChunks(checkedCsrCount(requests.size()), [&](CsrOffset offset, int count) {
        checkMpi(MPI_Waitall(count, requests.data() + offset, MPI_STATUSES_IGNORE),
                 "MPI_Waitall", communicator);
    });
}

CsrMatrix gatherCsrMatrix(const CsrMatrix& localResult, const std::vector<RowBlock>& blocks,
                          int resultCols, int rank, int ranks, MPI_Comm communicator) {
    if (rank == 0) {
        CsrMatrix globalResult;
        globalResult.rows = blocks.back().firstRow + blocks.back().rows;
        globalResult.cols = resultCols;
        globalResult.rowPtr.assign(static_cast<std::size_t>(globalResult.rows) + 1, 0);

        appendCsrBlock(globalResult, localResult, blocks[0], resultCols);

        for (int peer = 1; peer < ranks; ++peer) {
            std::array<CsrOffset, 3> header{};
            csrRecv(header.data(), 3, peer, kResultHeaderTag, communicator);

            CsrMatrix peerResult;
            peerResult.rows = checkedMatrixDimension(header[0]);
            peerResult.cols = checkedMatrixDimension(header[1]);
            peerResult.rowPtr.resize(static_cast<std::size_t>(peerResult.rows) + 1);
            peerResult.columnIndices.resize(csrSize<int>(header[2]));
            peerResult.values.resize(csrSize<double>(header[2]));

            csrRecv(dataOrNull(peerResult.rowPtr), checkedCsrCount(peerResult.rowPtr.size()), peer, kResultRowPtrTag, communicator);
            csrRecv(dataOrNull(peerResult.columnIndices), header[2], peer, kResultColumnTag, communicator);
            csrRecv(dataOrNull(peerResult.values), header[2], peer, kResultValueTag, communicator);
            appendCsrBlock(globalResult, peerResult, blocks[peer], resultCols);
        }
        return globalResult;
    }

    const std::array<CsrOffset, 3> header = {localResult.rows, localResult.cols,
                                       checkedCsrCount(localResult.values.size())};
    csrSend(header.data(), 3, 0, kResultHeaderTag, communicator);
    csrSend(dataOrNull(localResult.rowPtr), checkedCsrCount(localResult.rowPtr.size()), 0, kResultRowPtrTag, communicator);
    csrSend(dataOrNull(localResult.columnIndices), header[2], 0, kResultColumnTag, communicator);
    csrSend(dataOrNull(localResult.values), header[2], 0, kResultValueTag, communicator);
    return {};
}

CsrMatrix serialSpgemm(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    return buildSerialRowWiseProduct(matrixA, matrixB);
}

std::int64_t scalarMultiplicationCount(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    validateMultiplicationDimensions(matrixA, matrixB);
    std::int64_t count = 0;
    for (int row = 0; row < matrixA.rows; ++row) {
        for (CsrOffset entry = matrixA.rowPtr[row]; entry < matrixA.rowPtr[row + 1]; ++entry) {
            const int bRow = matrixA.columnIndices[entry];
            if (bRow < 0 || bRow >= matrixB.rows) {
                throw std::runtime_error("A contains a column index outside B's row range");
            }
            count = checkedCsrAdd(count, matrixB.rowPtr[bRow + 1] - matrixB.rowPtr[bRow], "scalar products");
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
        CsrOffset left = lhs.rowPtr[row];
        CsrOffset right = rhs.rowPtr[row];
        const CsrOffset leftEnd = lhs.rowPtr[row + 1];
        const CsrOffset rightEnd = rhs.rowPtr[row + 1];

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

const char* ValidationResult::status() const {
    if (mode == ValidationMode::None) return "SKIPPED";
    if (mode == ValidationMode::Partial) return passed ? "PARTIAL_PASS" : "PARTIAL_FAIL";
    return passed ? "PASS" : "FAIL";
}

PartialReference buildPartialReference(const CsrMatrix& matrixA, const CsrMatrix& matrixB) {
    validateMultiplicationDimensions(matrixA, matrixB);
    PartialReference reference{matrixA.rows, matrixB.cols, true, {}};
    reference.rowNnz.resize(static_cast<std::size_t>(matrixA.rows));
    forEachSerialProductRow(matrixA, matrixB, [&](int row, const auto& accumulator) {
        CsrOffset count = 0;
        for (const auto& entry : accumulator) {
            if (!std::isfinite(entry.second)) reference.finite = false;
            if (entry.second != 0.0) ++count;
        }
        reference.rowNnz[row] = count;
    });
    return reference;
}

ValidationResult comparePartialReference(const PartialReference& reference, const CsrMatrix& result) {
    ValidationResult validation;
    validation.mode = ValidationMode::Partial;
    if (result.rows != reference.rows || result.cols != reference.cols) {
        validation.passed = false;
        return validation;
    }
    const bool resultFinite = std::all_of(result.values.begin(), result.values.end(),
                                          [](double value) { return std::isfinite(value); });
    validation.passed = reference.finite && resultFinite;
    for (int row = 0; row < reference.rows; ++row) {
        if (reference.rowNnz.at(row) != result.rowPtr.at(row + 1) - result.rowPtr.at(row))
            ++validation.rowNnzMismatches;
    }
    validation.passed = validation.passed && validation.rowNnzMismatches == 0;
    return validation;
}

// BEGIN full-streaming-validation-v2
#include <cfenv>
#include <sstream>

BenchmarkMetrics validationMetadata(const ValidationResult& validation) {
    std::ostringstream ratio;
    ratio << std::setprecision(std::numeric_limits<long double>::max_digits10) << validation.maxErrorRatio;
    const bool full = validation.mode == ValidationMode::Full;
    return {{"validation_criterion", full ? "componentwise_dot_bound_v1" :
                                       validation.mode == ValidationMode::Partial ? "row_nnz_exact_v1" : "none"},
            {"validation_max_error_ratio", full ? ratio.str() : "NA"},
            {"validation_rejected_coordinates", full ? std::to_string(validation.rejectedCoordinates) : "NA"},
            {"validation_reference_digits", full ? std::to_string(validation.referenceDigits) : "NA"}};
}

namespace {
// The bound covers at most m products and m additions per output coordinate.
// For precision u and gradual-underflow unit eta:
//   |fl(dot)-dot| <= gamma_(2m)*sum|a*b| + 2m*eta/(1-2m*u).
// eta (rather than eta/2) is a conservative allowance for each operation.
// See docs/numerical-validation.md for the model, limitations and derivation.
long double upward(long double value) {
    return std::nextafter(value, std::numeric_limits<long double>::infinity());
}
long double addUp(long double a, long double b) {
    if (a == 0) return b;
    if (b == 0) return a;
    return upward(a + b);
}
long double multiplyUp(long double a, long double b) {
    return a == 0 || b == 0 ? 0 : upward(a * b);
}
long double divideUp(long double a, long double b) {
    return a == 0 ? 0 : upward(a / b);
}

template <typename T>
bool gradualUnderflow() {
    volatile T normal = std::numeric_limits<T>::min();
    volatile T half = T(0.5);
    volatile T subnormal = std::numeric_limits<T>::denorm_min();
    volatile T large = std::numeric_limits<T>::max();
    return normal * half == std::numeric_limits<T>::min() / T(2) &&
           subnormal * large == std::numeric_limits<T>::denorm_min() * std::numeric_limits<T>::max();
}

bool validNumericCsr(const CsrMatrix& matrix, bool ordered) {
    if (matrix.rows < 0 || matrix.cols < 0 ||
        matrix.rowPtr.size() != static_cast<std::size_t>(matrix.rows) + 1 ||
        matrix.rowPtr.front() != 0 || matrix.rowPtr.back() != checkedCsrCount(matrix.values.size()) ||
        matrix.columnIndices.size() != matrix.values.size()) return false;
    for (int row = 0; row < matrix.rows; ++row) {
        const auto first = matrix.rowPtr[row], end = matrix.rowPtr[row + 1];
        if (first < 0 || end < first || end > checkedCsrCount(matrix.values.size())) return false;
        int previous = -1;
        for (CsrOffset p = first; p < end; ++p) {
            const int column = matrix.columnIndices[p];
            if (column < 0 || column >= matrix.cols || (ordered && column <= previous) ||
                !std::isfinite(matrix.values[p])) return false;
            previous = column;
        }
    }
    return true;
}

struct NumericCell {
    double serial = 0;             // Historical TSV diagnostic; not the acceptance oracle.
    long double reference = 0;     // Same operands, evaluated in long double.
    long double magnitude = 0;     // Rounded sum of absolute long-double products.
    CsrOffset terms = 0;
};

struct ErrorModel {
    long double gamma;
    long double underflow;
};

template <typename T>
ErrorModel errorModel(CsrOffset terms) {
    const long double operations = multiplyUp(2, static_cast<long double>(terms));
    const long double scaled = multiplyUp(operations, std::numeric_limits<T>::epsilon() / 2.0L);
    // Fail closed when inversion of the magnitude estimate cannot be justified.
    if (!(scaled < 0.5L)) return {1, std::numeric_limits<long double>::infinity()};
    const long double denominator = std::nextafter(1 - scaled, 0.0L);
    return {divideUp(scaled, denominator),
            divideUp(multiplyUp(operations, std::numeric_limits<T>::denorm_min()), denominator)};
}

long double coordinateBound(const NumericCell& cell) {
    if (cell.terms == 0) return 0;
    const auto candidate = errorModel<double>(cell.terms);
    const auto reference = errorModel<long double>(cell.terms);
    if (!(candidate.gamma < 1 && reference.gamma < 1) ||
        !std::isfinite(cell.reference) || !std::isfinite(cell.magnitude))
        return std::numeric_limits<long double>::infinity();
    // |magnitude - S| <= gamma_L*S + underflow_L, hence this upper bound on S.
    const long double scale = divideUp(addUp(cell.magnitude, reference.underflow),
                                       std::nextafter(1 - reference.gamma, 0.0L));
    return addUp(multiplyUp(addUp(candidate.gamma, reference.gamma), scale),
                 addUp(candidate.underflow, reference.underflow));
}
}  // namespace

static ValidationResult validateFullStreaming(const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                                               const CsrMatrix& result) {
    ValidationResult validation;
    validation.maxAbsoluteError = std::numeric_limits<double>::infinity();
    validation.passed = false;
    validation.referenceDigits = std::numeric_limits<long double>::digits;
    std::cout << "validation_full_strategy=streaming_roundoff_v2"
              << " validation_criterion=componentwise_dot_bound_v1"
              << " validation_reference_digits=" << std::numeric_limits<long double>::digits
              << " validation_candidate_digits=" << std::numeric_limits<double>::digits << '\n';
    // The bound assumes binary round-to-nearest with gradual underflow, without reassociation.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    std::cout << "validation_invalid_environment=fast_math\n";
    return validation;
#endif
    if (std::numeric_limits<double>::radix != 2 || std::numeric_limits<long double>::radix != 2 ||
        std::numeric_limits<double>::digits != 53 ||
        std::numeric_limits<long double>::digits < std::numeric_limits<double>::digits ||
        std::numeric_limits<double>::has_denorm != std::denorm_present ||
        std::numeric_limits<long double>::has_denorm != std::denorm_present ||
        std::fegetround() != FE_TONEAREST || !gradualUnderflow<double>() ||
        !gradualUnderflow<long double>()) {
        std::cout << "validation_invalid_environment=unsupported_floating_point\n";
        return validation;
    }
    if (!validNumericCsr(matrixA, false) || !validNumericCsr(matrixB, false) ||
        !validNumericCsr(result, true) || matrixA.cols != matrixB.rows ||
        result.rows != matrixA.rows || result.cols != matrixB.cols) {
        std::cout << "validation_invalid_matrix=1\n";
        return validation;
    }
    const auto start = std::chrono::steady_clock::now();
    double maximumSerialError = 0;
    long double maximumReferenceError = 0, maximumRatio = 0;
    long double worstActual = 0, worstReference = 0, worstBound = 0;
    int worstRow = -1, worstColumn = -1, examples = 0;
    CsrOffset supportMismatches = 0, referenceNnz = 0, rejected = 0;
    std::unordered_map<int, NumericCell> accumulator;
    std::vector<int> columns;
    for (int row = 0; row < matrixA.rows; ++row) {
        accumulator.clear();
        for (CsrOffset a = matrixA.rowPtr[row]; a < matrixA.rowPtr[row + 1]; ++a) {
            const int bRow = matrixA.columnIndices[a];
            for (CsrOffset b = matrixB.rowPtr[bRow]; b < matrixB.rowPtr[bRow + 1]; ++b) {
                auto& cell = accumulator[matrixB.columnIndices[b]];
                const double product = matrixA.values[a] * matrixB.values[b];
                const long double precise = static_cast<long double>(matrixA.values[a]) *
                                            static_cast<long double>(matrixB.values[b]);
                cell.serial += product;
                cell.reference += precise;
                cell.magnitude += std::abs(precise);
                cell.terms = checkedCsrAdd(cell.terms, 1, "validation contributions");
            }
        }
        columns.clear();
        CsrOffset rowNnz = 0;
        for (const auto& entry : accumulator) {
            columns.push_back(entry.first); // Keep cancelled coordinates for the numerical comparison.
            if (entry.second.serial != 0) ++rowNnz;
        }
        std::sort(columns.begin(), columns.end());
        referenceNnz = checkedCsrAdd(referenceNnz, rowNnz, "validation reference nnz");
        CsrOffset actual = result.rowPtr[row];
        const auto actualEnd = result.rowPtr[row + 1];
        if (actualEnd - actual != rowNnz) ++validation.rowNnzMismatches;
        std::size_t expected = 0;
        while (actual < actualEnd || expected < columns.size()) {
            NumericCell cell;
            double actualValue = 0;
            bool actualStored = false;
            int column;
            if (expected == columns.size() ||
                (actual < actualEnd && result.columnIndices[actual] < columns[expected])) {
                column = result.columnIndices[actual];
                actualValue = result.values[actual++];
                actualStored = true;
            } else {
                column = columns[expected++];
                cell = accumulator.at(column);
                if (actual < actualEnd && result.columnIndices[actual] == column) {
                    actualValue = result.values[actual++];
                    actualStored = true;
                }
            }
            const bool supportMismatch = actualStored != (cell.serial != 0);
            double serialError = std::abs(actualValue - cell.serial);
            if (!std::isfinite(serialError)) serialError = std::numeric_limits<double>::infinity();
            maximumSerialError = std::max(maximumSerialError, serialError);
            const long double difference = std::abs(static_cast<long double>(actualValue) - cell.reference);
            const long double errorUpper = difference == 0 ? 0 : upward(difference);
            const long double bound = coordinateBound(cell);
            // An infinite/unavailable bound is never permission to accept a coordinate.
            const bool accepted = std::isfinite(serialError) && std::isfinite(errorUpper) &&
                                  std::isfinite(bound) && errorUpper <= bound;
            const long double ratio = !std::isfinite(serialError) || !std::isfinite(errorUpper) || !std::isfinite(bound) ||
                                      (bound == 0 && difference != 0)
                ? std::numeric_limits<long double>::infinity()
                : bound == 0 ? 0 : accepted ? std::min(1.0L, divideUp(errorUpper, bound))
                                           : divideUp(errorUpper, bound);
            if (ratio > maximumRatio) {
                maximumRatio = ratio;
                worstRow = row;
                worstColumn = column;
                worstActual = actualValue;
                worstReference = cell.reference;
                worstBound = bound;
            }
            maximumReferenceError = std::max(maximumReferenceError,
                std::isfinite(difference) ? difference : std::numeric_limits<long double>::infinity());
            if (!accepted) ++rejected;
            if (supportMismatch) ++supportMismatches;
            if (examples < 8 && (supportMismatch || !accepted)) {
                std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10)
                          << "validation_difference row=" << row << " column=" << column
                          << " actual=" << actualValue << " expected=" << cell.reference
                          << " serial_expected=" << cell.serial << " abs_error=" << difference
                          << " error_bound=" << bound << " error_ratio=" << ratio
                          << " contributions=" << cell.terms << " accepted=" << accepted
                          << " support_mismatch=" << supportMismatch << '\n';
                ++examples;
            }
        }
    }
    validation.maxAbsoluteError = maximumSerialError;
    validation.passed = rejected == 0;
    validation.maxErrorRatio = maximumRatio;
    validation.rejectedCoordinates = rejected;
    std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10)
              << "validation_index_base=0 validation_full_rows=" << matrixA.rows
              << " validation_reference_nnz=" << referenceNnz
              << " validation_row_nnz_mismatches=" << validation.rowNnzMismatches
              << " validation_support_mismatches=" << supportMismatches
              << " validation_rejected_coordinates=" << rejected
              << " validation_max_abs_error=" << maximumSerialError
              << " validation_max_reference_error=" << maximumReferenceError
              << " validation_max_error_ratio=" << maximumRatio
              << " validation_worst_ratio_row=" << worstRow
              << " validation_worst_ratio_column=" << worstColumn
              << " validation_worst_ratio_actual=" << worstActual
              << " validation_worst_ratio_reference=" << worstReference
              << " validation_worst_ratio_bound=" << worstBound
              << " validation_full_seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
    return validation;
}
// END full-streaming-validation-v2

ValidationResult validateProduct(const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                                 const CsrMatrix& result, ValidationMode mode,
                                 const std::string& partialCacheDirectory) {
    ValidationResult validation;
    validation.mode = mode;
    if (mode == ValidationMode::None) return validation;
    if (mode == ValidationMode::Full) {
        return validateFullStreaming(matrixA, matrixB, result);
    }
    validateMultiplicationDimensions(matrixA, matrixB);
    if (result.rows != matrixA.rows || result.cols != matrixB.cols) {
        validation.passed = false;
        return validation;
    }
    using Clock = std::chrono::steady_clock;
    const auto seconds = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    ValidationDiagnostics diagnostics;
    PartialReference reference;
    auto start = Clock::now();
    if (partialCacheDirectory.empty()) {
        diagnostics.source = "regenerated";
        reference = buildPartialReference(matrixA, matrixB);
        diagnostics.buildSeconds = seconds(start);
    } else {
        diagnostics.source = "reused";
        diagnostics.key = partialReferenceKey(matrixA, matrixB);
        diagnostics.identitySeconds = seconds(start);
        start = Clock::now();
        reference = readPartialReference(partialCacheDirectory, diagnostics.key, matrixA.rows, matrixB.cols);
        diagnostics.readSeconds = seconds(start);
    }
    start = Clock::now();
    validation = comparePartialReference(reference, result);
    diagnostics.compareSeconds = seconds(start);
    printValidationDiagnostics(diagnostics);
    return validation;
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

std::string timingSamplesJson(const std::vector<double>& samples) {
    std::ostringstream output;
    output << std::setprecision(17) << '[';
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) output << ',';
        output << samples[i];
    }
    output << ']';
    return output.str();
}

void appendBenchmarkResult(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                           const CsrMatrix& matrixC, double distributionSeconds,
                           double haloSetupSeconds, double firstProductSeconds,
                           double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double gflops,
                           const ValidationResult& validation,
                           const std::vector<double>& communicationSamples,
                           const std::vector<double>& computeSamples,
                           const std::vector<double>& endToEndSamples,
                           double fullProductSeconds, double firstGatherSeconds,
                           const BenchmarkMetrics& metrics) {
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

    std::string header =
        "implementation\texperiment\tmatrix_a_source\tmatrix_b_source"
        "\ta_rows\ta_cols\tb_rows\tb_cols\ta_nnz\tb_nnz\tc_nnz\tranks"
        "\tthreads_per_rank\tomp_schedule\tomp_chunk\twarmup\trepeats\ttrials"
        "\tdistribution_seconds\thalo_setup_seconds\tfirst_product_seconds"
        "\tcommunication_p90_seconds\tcompute_p90_seconds\tend_to_end_p90_seconds"
        "\tgather_seconds\tcompute_gflops_p90\tmax_abs_error\tvalidation\tbenchmark_protocol"
        "\tcommunication_samples_seconds\tcompute_samples_seconds\tend_to_end_samples_seconds"
        "\tfull_product_seconds\tfirst_gather_seconds";
    for (const auto& metric : metrics) header += '\t' + metric.first;
    // BEGIN roundoff-validation-metadata-v1
    const auto validationFields = validationMetadata(validation);
    for (const auto& field : validationFields) header += '\t' + field.first;
    // END roundoff-validation-metadata-v1
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
    if (validation.maxAbsoluteError) {
        output << *validation.maxAbsoluteError;
    } else {
        output << "NA";
    }
    output << '\t' << validation.status() << "\tprepared_halo_csr64_v3\t"
           << timingSamplesJson(communicationSamples) << '\t'
           << timingSamplesJson(computeSamples) << '\t'
           << timingSamplesJson(endToEndSamples) << '\t'
           << fullProductSeconds << '\t' << firstGatherSeconds;
    for (const auto& metric : metrics) output << '\t' << metric.second;
    // BEGIN roundoff-validation-metadata-v1
    for (const auto& field : validationFields) output << '\t' << field.second;
    // END roundoff-validation-metadata-v1
    output << '\n';
    output.flush();
    if (!output) throw std::runtime_error("failed writing results file: " + outputPath.string());
}

void printBenchmarkSummary(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                           const CsrMatrix& matrixC, double distributionSeconds,
                           double haloSetupSeconds, double firstProductSeconds,
                           double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double computeGflops,
                           const ValidationResult& validation,
                           double fullProductSeconds, double firstGatherSeconds) {
    std::cout << std::fixed << std::setprecision(6) << "implementation=" << implementation << '\n'
              << "benchmark_protocol=prepared_halo_csr64_v3\n"
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
              << "full_product_seconds=" << fullProductSeconds << '\n'
              << "first_gather_seconds=" << firstGatherSeconds << '\n'
              << "compute_gflops=" << computeGflops << '\n'
              << "max_abs_error=";
    if (validation.maxAbsoluteError) {
        std::cout << *validation.maxAbsoluteError;
    } else {
        std::cout << "NA";
    }
    std::cout << '\n'
              << "validation=" << validation.status() << '\n'
              << "results_file=" << options.resultsPath << '\n';
}
