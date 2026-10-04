#include "spgemm_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {

template <typename T>
std::string number(T value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

// Population statistics, including empty rows/ranks, without a temporary row array.
struct Statistics {
    std::int64_t count = 0, zeros = 0, minimum = 0, maximum = 0;
    long double mean = 0, m2 = 0;

    void add(std::int64_t value) {
        minimum = count == 0 ? value : std::min(minimum, value);
        maximum = std::max(maximum, value);
        zeros += value == 0;
        ++count;
        const long double delta = value - mean;
        mean += delta / count;
        m2 += delta * (value - mean);
    }
    long double stddev() const { return count ? std::sqrt(std::max(0.0L, m2 / count)) : 0; }
    long double cv() const { return mean > 0 ? stddev() / mean : 0; }
};

void appendRowMetrics(BenchmarkMetrics& metrics, const std::string& prefix, const CsrMatrix& matrix) {
    Statistics stats;
    for (int row = 0; row < matrix.rows; ++row) stats.add(matrix.rowPtr[row + 1] - matrix.rowPtr[row]);
    metrics.emplace_back(prefix + "_row_nnz_min", number(stats.minimum));
    metrics.emplace_back(prefix + "_row_nnz_max", number(stats.maximum));
    metrics.emplace_back(prefix + "_mean_nnz_per_row", number(stats.mean));
    metrics.emplace_back(prefix + "_row_nnz_stddev", number(stats.stddev()));
    metrics.emplace_back(prefix + "_row_nnz_cv", number(stats.cv()));
    metrics.emplace_back(prefix + "_empty_rows", number(stats.zeros));
}

std::string countsJson(const std::vector<std::int64_t>& values) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) stream << ',';
        stream << values[i];  // Preserve exact integers, including counts above 2^53.
    }
    stream << ']';
    return stream.str();
}

}  // namespace

std::vector<std::int64_t> rowPartitionWork(const CsrMatrix& a, const CsrMatrix& b, int ranks) {
    std::vector<std::int64_t> work(ranks, 0);
    const auto blocks = partitionRows(a.rows, ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        const auto block = blocks[rank];
        for (int row = block.firstRow; row < block.firstRow + block.rows; ++row) {
            for (int p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p) {
                const int inner = a.columnIndices[p];
                work[rank] += b.rowPtr[inner + 1] - b.rowPtr[inner];
            }
        }
    }
    return work;
}

BenchmarkMetrics collectBenchmarkMetrics(
    const CsrMatrix& globalA, const CsrMatrix& globalB, const CsrMatrix& globalC,
    const CsrMatrix& localA, const CsrMatrix& localB, const CsrMatrix& localC,
    const std::vector<std::int64_t>& work, MPI_Comm communicator) {
    int rank = 0, ranks = 0;
    checkMpi(MPI_Comm_rank(communicator, &rank), "MPI_Comm_rank(metrics)", communicator);
    checkMpi(MPI_Comm_size(communicator, &ranks), "MPI_Comm_size(metrics)", communicator);
    const std::array<std::int64_t, 3> local{
        static_cast<std::int64_t>(localA.values.size()), static_cast<std::int64_t>(localB.values.size()),
        static_cast<std::int64_t>(localC.values.size())};
    std::vector<std::array<std::int64_t, 3>> counts(rank == 0 ? ranks : 0);
    checkMpi(MPI_Gather(local.data(), 3, MPI_INT64_T, counts.data(), 3, MPI_INT64_T, 0, communicator),
             "MPI_Gather(structure metrics)", communicator);
    if (rank != 0) return {};
    const auto totalWork = scalarMultiplicationCount(globalA, globalB);
    if (work.size() != static_cast<std::size_t>(ranks) ||
        std::accumulate(work.begin(), work.end(), std::int64_t{0}) != totalWork) {
        throw std::runtime_error("rank workload does not match the global scalar-product count");
    }
    BenchmarkMetrics metrics;
    appendRowMetrics(metrics, "a", globalA);
    appendRowMetrics(metrics, "b", globalB);
    appendRowMetrics(metrics, "c", globalC);
    metrics.emplace_back("result_finite", std::all_of(globalC.values.begin(), globalC.values.end(),
                         [](double value) { return std::isfinite(value); }) ? "1" : "0");
    const std::array<std::size_t, 3> globalNnz{globalA.values.size(), globalB.values.size(), globalC.values.size()};
    const std::array<std::string, 3> names{"rank_a_nnz", "rank_b_nnz", "rank_c_nnz"};
    for (std::size_t matrix = 0; matrix < names.size(); ++matrix) {
        std::vector<std::int64_t> values;
        for (const auto& owned : counts) values.push_back(owned[matrix]);
        if (std::accumulate(values.begin(), values.end(), std::int64_t{0}) !=
            static_cast<std::int64_t>(globalNnz[matrix])) {
            throw std::runtime_error("rank nnz counts do not match the global matrix");
        }
        metrics.emplace_back(names[matrix], countsJson(values));
    }
    Statistics stats;
    for (const auto value : work) stats.add(value);
    metrics.emplace_back("rank_scalar_products", countsJson(work));
    metrics.emplace_back("scalar_products", number(totalWork));
    metrics.emplace_back("rank_work_min", number(stats.minimum));
    metrics.emplace_back("rank_work_max", number(stats.maximum));
    metrics.emplace_back("rank_work_mean", number(stats.mean));
    metrics.emplace_back("rank_work_stddev", number(stats.stddev()));
    metrics.emplace_back("rank_work_cv", number(stats.cv()));
    metrics.emplace_back("rank_work_max_over_mean", number(stats.mean > 0 ? stats.maximum / stats.mean : 0));
    metrics.emplace_back("idle_ranks", number(stats.zeros));
    return metrics;
}
