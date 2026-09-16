#pragma once

#include <mpi.h>
#include <omp.h>

#include "csr_matrix.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
    bool validate = true;
    std::string schedule = "guided";
    std::string matrixAPath;
    std::string matrixBPath;
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
double maxRankValue(double value, MPI_Comm communicator);
double maxElapsed(double start, MPI_Comm communicator);
double percentile90(std::vector<double> samples);

// An absent maximum error denotes skipped validation in both output formats.
void appendBenchmarkResult(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                           const CsrMatrix& matrixC, double distributionSeconds,
                           double haloSetupSeconds, double firstProductSeconds,
                           double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double gflops,
                           std::optional<double> maxAbsoluteError);
void printBenchmarkSummary(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrixA, const CsrMatrix& matrixB,
                           const CsrMatrix& matrixC, double distributionSeconds,
                           double haloSetupSeconds, double firstProductSeconds,
                           double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double computeGflops,
                           std::optional<double> maxAbsoluteError);
