#include "benchmark.hpp"

#include "matrix_market.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace trident {
namespace {

struct RunOptions {
    Options benchmark;
    int logicalNodeSize = 0;
};

RunOptions parseArguments(int argc, char** argv, const std::string& implementation) {
    RunOptions result;
    std::vector<char*> commonArgs{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--logical-node-size") {
            if (++i == argc) throw std::invalid_argument("missing --logical-node-size value");
            const std::string value = argv[i];
            std::size_t used = 0;
            result.logicalNodeSize = std::stoi(value, &used);
            if (used != value.size() || result.logicalNodeSize < 1) {
                throw std::invalid_argument("--logical-node-size expects a positive integer");
            }
        } else {
            if (argument == "--help") {
                std::cout << "Trident CPU: square grid of nodes, equal MPI ranks per node.\n"
                             "  --logical-node-size N  Single-host topology emulation for correctness tests only\n";
            }
            commonArgs.push_back(argv[i]);
        }
    }
    result.benchmark = parseOptions(mpiCount(commonArgs.size()), commonArgs.data(), implementation,
                                    "results/trident/" + implementation + "_v1.tsv");
    return result;
}

template <typename T>
std::string number(T value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

using Record = std::vector<std::pair<std::string, std::string>>;

void writeRecord(const Record& record, const std::string& path) {
    std::ostringstream header;
    std::ostringstream row;
    for (std::size_t i = 0; i < record.size(); ++i) {
        if (i != 0) { header << '\t'; row << '\t'; }
        header << record[i].first;
        row << record[i].second;
    }
    const std::filesystem::path outputPath(path);
    if (outputPath.has_parent_path()) std::filesystem::create_directories(outputPath.parent_path());
    const bool newFile = !std::filesystem::exists(outputPath) || std::filesystem::file_size(outputPath) == 0;
    if (!newFile) {
        std::ifstream input(outputPath);
        std::string existing;
        std::getline(input, existing);
        if (!existing.empty() && existing.back() == '\r') existing.pop_back();
        if (existing != header.str()) {
            throw std::runtime_error("incompatible benchmark TSV header; choose a new --results path: " + path);
        }
    }
    std::ofstream output(outputPath, std::ios::app);
    if (!output) throw std::runtime_error("cannot write results file: " + path);
    if (newFile) output << header.str() << '\n';
    output << row.str() << '\n';
    output.flush();
    if (!output) throw std::runtime_error("failed writing results file: " + path);
}

}  // namespace

int runBenchmark(int argc, char** argv, const std::string& implementation, ExchangeFactory factory) {
    RunOptions run;
    // Parse before MPI initialization so --help can use the existing common parser's normal exit.
    try {
        run = parseArguments(argc, argv, implementation);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed\n";
        return EXIT_FAILURE;
    }
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required", MPI_COMM_WORLD);
    int exitCode = EXIT_SUCCESS;
    try {
        const auto& options = run.benchmark;
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        omp_set_schedule(parseSchedule(options.schedule), options.chunk);
        ProcessGrid grid(MPI_COMM_WORLD, run.logicalNodeSize);
        CsrMatrix globalA;
        CsrMatrix globalB;
        std::array<int, 4> shape{};
        if (grid.rank == 0) {
            globalA = options.matrixAPath.empty()
                          ? makeBandedMatrix(options.rows, options.cols, options.nonZerosPerRow)
                          : readMatrixMarket(options.matrixAPath);
            if (!options.matrixBPath.empty()) {
                globalB = readMatrixMarket(options.matrixBPath);
            } else if (!options.matrixAPath.empty()) {
                if (globalA.rows != globalA.cols) {
                    throw std::invalid_argument(
                        "--matrix-a without --matrix-b requires a square A so it can be reused as B");
                }
                globalB = globalA;
            } else {
                globalB = makeBandedMatrix(options.cols, options.bCols, options.bNonZerosPerRow);
            }
            if (globalA.cols != globalB.rows) throw std::invalid_argument("SpGEMM requires A.cols == B.rows");
            shape = {globalA.rows, globalA.cols, globalB.rows, globalB.cols};
        }
        checkMpi(MPI_Bcast(shape.data(), 4, MPI_INT, 0, grid.world), "MPI_Bcast(shape)", grid.world);
        checkMpi(MPI_Barrier(grid.world), "MPI_Barrier(distribution)", grid.world);
        const double distributionStart = MPI_Wtime();
        const auto a = distributeBlocks(grid.rank == 0 ? &globalA : nullptr, shape[0], shape[1], grid);
        const auto b = distributeBlocks(grid.rank == 0 ? &globalB : nullptr, shape[2], shape[3], grid);
        const double distributionSeconds = maxElapsed(distributionStart, grid.world);

        checkMpi(MPI_Barrier(grid.world), "MPI_Barrier(setup)", grid.world);
        const double setupStart = MPI_Wtime();
        const auto plan = preparePlan(a, b, grid);
        ProductWorkspace workspace(plan);
        auto exchange = factory(grid, plan);
        checkMpi(MPI_Barrier(grid.world), "MPI_Barrier(setup complete)", grid.world);
        const double setupEnd = MPI_Wtime();
        auto product = multiply(a, b, grid, plan, *exchange, workspace);
        const double firstEnd = MPI_Wtime();
        const double setupSeconds = maxRankValue(setupEnd - setupStart, grid.world);
        const double firstSeconds = maxRankValue(firstEnd - setupStart, grid.world);

        for (int i = 0; i < options.warmup; ++i) {
            checkMpi(MPI_Barrier(grid.world), "MPI_Barrier(warmup)", grid.world);
            product = multiply(a, b, grid, plan, *exchange, workspace);
        }
        std::array<std::vector<double>, 5> samples;
        for (int trial = 0; trial < options.trials; ++trial) {
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                checkMpi(MPI_Barrier(grid.world), "MPI_Barrier(sample)", grid.world);
                const double sampleStart = MPI_Wtime();
                product = multiply(a, b, grid, plan, *exchange, workspace);
                const double sampleSeconds = MPI_Wtime() - sampleStart;
                const std::array<double, 5> local{
                    product.interNodeSeconds, product.intraNodeSeconds,
                    product.interNodeSeconds + product.intraNodeSeconds,
                    product.computeSeconds, sampleSeconds};
                std::array<double, 5> maxima{};
                checkMpi(MPI_Reduce(local.data(), maxima.data(), 5, MPI_DOUBLE, MPI_MAX, 0, grid.world),
                         "MPI_Reduce(sample timings)", grid.world);
                if (grid.rank == 0) {
                    for (std::size_t i = 0; i < maxima.size(); ++i) samples[i].push_back(maxima[i]);
                }
            }
        }
        checkMpi(MPI_Barrier(grid.world), "MPI_Barrier(gather)", grid.world);
        const double gatherStart = MPI_Wtime();
        const auto result = gatherBlocks(product.matrix, shape[0], shape[3], grid);
        const double gatherSeconds = maxElapsed(gatherStart, grid.world);
        if (grid.rank == 0) {
            std::optional<double> error;
            if (options.validate) error = maxAbsoluteDifference(result, serialSpgemm(globalA, globalB));
            const std::string validation = !error ? "SKIPPED" : validationPassed(*error) ? "PASS" : "FAIL";
            exitCode = validation == "FAIL" ? EXIT_FAILURE : EXIT_SUCCESS;
            std::array<double, 5> p90{};
            for (std::size_t i = 0; i < p90.size(); ++i) p90[i] = percentile90(samples[i]);
            const double flops = 2.0 * static_cast<double>(scalarMultiplicationCount(globalA, globalB));
            const std::string aSource = options.matrixAPath.empty() ? "synthetic" : options.matrixAPath;
            const std::string bSource = options.matrixBPath.empty() ? aSource : options.matrixBPath;
            Record record{
                {"implementation", implementation}, {"benchmark_protocol", "trident_staged_csr_v1"},
                {"experiment", options.experiment}, {"matrix_a_source", aSource}, {"matrix_b_source", bSource},
                {"a_rows", number(shape[0])}, {"a_cols", number(shape[1])},
                {"b_rows", number(shape[2])}, {"b_cols", number(shape[3])},
                {"a_nnz", number(globalA.values.size())}, {"b_nnz", number(globalB.values.size())},
                {"c_nnz", number(result.values.size())}, {"ranks", number(grid.size)},
                {"threads_per_rank", number(options.threads)}, {"omp_schedule", options.schedule},
                {"omp_chunk", number(options.chunk)}, {"warmup", number(options.warmup)},
                {"repeats", number(options.repeats)}, {"trials", number(options.trials)},
                {"nodes", number(grid.nodeCount)}, {"ranks_per_node", number(grid.nodeSize)},
                {"grid_rows", number(grid.side)}, {"grid_cols", number(grid.side)},
                {"topology", grid.emulated ? "logical_test" : "physical"},
                {"inter_node_transport", "two_sided_static_cannon"},
                {"intra_node_transport", implementation},
                {"distribution_seconds", number(distributionSeconds)},
                {"plan_setup_seconds", number(setupSeconds)}, {"first_product_seconds", number(firstSeconds)},
                {"inter_node_p90_seconds", number(p90[0])}, {"intra_node_p90_seconds", number(p90[1])},
                {"communication_p90_seconds", number(p90[2])}, {"compute_p90_seconds", number(p90[3])},
                {"end_to_end_p90_seconds", number(p90[4])}, {"gather_seconds", number(gatherSeconds)},
                {"compute_gflops", number(p90[3] > 0.0 ? flops / p90[3] / 1.0e9 : 0.0)},
                {"max_abs_error", error ? number(*error) : "NA"}, {"validation", validation}};
            writeRecord(record, options.resultsPath);
            for (const auto& field : record) std::cout << field.first << '=' << field.second << '\n';
            std::cout << "A=" << shape[0] << 'x' << shape[1] << " nnz=" << globalA.values.size() << '\n'
                      << "B=" << shape[2] << 'x' << shape[3] << " nnz=" << globalB.values.size() << '\n'
                      << "C=" << shape[0] << 'x' << shape[3] << " nnz=" << result.values.size() << '\n'
                      << "results_file=" << options.resultsPath << '\n';
        }
        checkMpi(MPI_Bcast(&exitCode, 1, MPI_INT, 0, grid.world), "MPI_Bcast(validation)", grid.world);
        exchange.reset();
        grid.close();
    } catch (const std::exception& error) {
        fail(error.what(), MPI_COMM_WORLD);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", MPI_COMM_WORLD);
    return exitCode;
}

}  // namespace trident
