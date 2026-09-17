#pragma once

#include "execution.hpp"

#include <memory>
#include <string>

namespace trident {

using ExchangeFactory = std::unique_ptr<IntraNodeExchange> (*)(const ProcessGrid&, const ExecutionPlan&);
using InterNodeFactory = std::unique_ptr<InterNodeExchange> (*)(const ProcessGrid&, const ExecutionPlan&,
                                                              const CsrMatrix&, const CsrMatrix&);

struct BenchmarkBackend {
    std::string implementation;
    ExchangeFactory intraFactory;
    InterNodeFactory interFactory;
    int threadLevel = MPI_THREAD_FUNNELED;
    std::string protocol = "trident_staged_csr_v1";
    std::string interTransport = "two_sided_static_cannon";
    std::string intraTransport = "trident_two_sided";
};

int runBenchmark(int argc, char** argv, const BenchmarkBackend& backend);

}  // namespace trident
