#pragma once

#include <mpi.h>
#include <omp.h>

#include "csr_matrix.hpp"

#include <string>
#include <vector>

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

double denseValue(int row, int column, int epoch = 0);
void fillOwnedDenseRows(RowBlock block, int denseCols, int epoch, std::vector<double>& values);
std::vector<double> makeOwnedDenseRows(RowBlock block, int denseCols, int epoch = 0);

void waitAll(std::vector<MPI_Request>& requests, MPI_Comm communicator);

std::vector<double> gatherResult(const std::vector<double>& localResult,
                                 const std::vector<RowBlock>& blocks, int denseCols, int rank,
                                 int ranks, MPI_Comm communicator);
std::vector<double> serialSpmm(const CsrMatrix& matrix, int denseCols, int denseEpoch);
double maxAbsoluteDifference(const std::vector<double>& lhs, const std::vector<double>& rhs);
double maxElapsed(double start, MPI_Comm communicator);
double percentile90(std::vector<double> samples);

void appendBenchmarkResult(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrix, double distributionSeconds,
                           double haloSetupSeconds, double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double gflops, double maxAbsoluteError);
void printBenchmarkSummary(const std::string& implementation, const Options& options, int ranks,
                           const CsrMatrix& matrix, double distributionSeconds,
                           double haloSetupSeconds, double communicationP90Seconds,
                           double computeP90Seconds, double endToEndP90Seconds,
                           double gatherSeconds, double computeGflops, double maxAbsoluteError);
