#pragma once

#include <mpi.h>
#include <omp.h>

#include "csr_matrix.hpp"
#include "mpi_large.hpp"
#include "partial_reference.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

enum class ValidationMode { Full, Partial, None };

struct ValidationResult {
    ValidationMode mode = ValidationMode::Full;
    bool passed = true;
    std::optional<double> maxAbsoluteError;
    int rowNnzMismatches = 0;

    // BEGIN roundoff-validation-metadata-v1
    long double maxErrorRatio = 0;
    std::int64_t rejectedCoordinates = 1;
    int referenceDigits = 0;
    // END roundoff-validation-metadata-v1

    const char* status() const;
};

struct Options {
    int rows = 1'024;
    int cols = 1'024;
    int bCols = 1'024;
    int nonZerosPerRow = 16;
    int bNonZerosPerRow = 16;
    int threads = 1;
    int warmup = 2;
    int repeats = 10;
    int trials = 5;
    int chunk = 64;
    ValidationMode validationMode = ValidationMode::Full;
    std::string schedule = "guided";
    std::string matrixAPath;
    std::string matrixBPath;
    std::string partialCacheDirectory;
    // BEGIN full-reference-cache-v1
    std::string fullCacheDirectory;
    // END full-reference-cache-v1
    std::string resultsPath;
    std::string experiment = "manual";
};

struct RowBlock {
    int firstRow = 0;
    int rows = 0;
};

[[noreturn]] void fail(const std::string& message, MPI_Comm communicator);
void checkMpi(int error, const char* call, MPI_Comm communicator);

Options parseOptions(int argc, char** argv, const std::string& programName,
                     const std::string& defaultResultsPath);
omp_sched_t parseSchedule(const std::string& schedule);

std::vector<RowBlock> partitionRows(int totalRows, int ranks);
int ownerOfRow(int row, const std::vector<RowBlock>& blocks);

CsrMatrix makeBandedMatrix(int rows, int cols, int nonZerosPerRow);
CsrMatrix sliceRows(const CsrMatrix& matrix, RowBlock block);
CsrMatrix distributeMatrix(const CsrMatrix* globalMatrix, const std::vector<RowBlock>& blocks,
                           int rank, int ranks, MPI_Comm communicator);

void waitAll(std::vector<MPI_Request>& requests, MPI_Comm communicator);

CsrMatrix gatherCsrMatrix(const CsrMatrix& localResult, const std::vector<RowBlock>& blocks,
                          int resultCols, int rank, int ranks, MPI_Comm communicator);
CsrMatrix serialSpgemm(const CsrMatrix& matrixA, const CsrMatrix& matrixB);
std::int64_t scalarMultiplicationCount(const CsrMatrix& matrixA, const CsrMatrix& matrixB);
// Returns infinity for incompatible shapes or non-finite values/differences.
double maxAbsoluteDifference(const CsrMatrix& lhs, const CsrMatrix& rhs);
bool validationPassed(double maxAbsoluteError);
// Partial validation computes one serial row at a time, drops exact zeros and compares
// row nnz counts. It rejects non-finite results but does not compare columns or values.
ValidationResult validateProduct(const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                                 const CsrMatrix& result, ValidationMode mode,
                                 const std::string& partialCacheDirectory = "",
                                 const std::string& fullCacheDirectory = "");
ValidationResult comparePartialReference(const PartialReference& reference, const CsrMatrix& result);
double maxRankValue(double value, MPI_Comm communicator);
double maxElapsed(double start, MPI_Comm communicator);
double percentile90(std::vector<double> samples);
// JSON array in trial-major, repetition-minor order; no samples are discarded.
std::string timingSamplesJson(const std::vector<double>& samples);

using BenchmarkMetrics = std::vector<std::pair<std::string, std::string>>;
// BEGIN roundoff-validation-metadata-v1
BenchmarkMetrics validationMetadata(const ValidationResult& validation);
// END roundoff-validation-metadata-v1
// Structural scalar products assigned to the baseline's contiguous row partitions.
std::vector<std::int64_t> rowPartitionWork(const CsrMatrix& a, const CsrMatrix& b, int ranks);
// Collective, outside measured regions. Global matrices and work are used only on rank 0.
// Rank arrays follow communicator rank order; nnz counts exclude halo/staging buffers.
BenchmarkMetrics collectBenchmarkMetrics(
    const CsrMatrix& globalA, const CsrMatrix& globalB, const CsrMatrix& globalC,
    const CsrMatrix& localA, const CsrMatrix& localB, const CsrMatrix& localC,
    const std::vector<std::int64_t>& work, MPI_Comm communicator);

// The numerical error is absent for both partial and skipped validation.
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
                           const BenchmarkMetrics& metrics);
void printBenchmarkSummary(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                           const CsrMatrix& matrixC, double distributionSeconds,
                           double haloSetupSeconds, double firstProductSeconds,
                           double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double computeGflops,
                           const ValidationResult& validation,
                           double fullProductSeconds, double firstGatherSeconds);
