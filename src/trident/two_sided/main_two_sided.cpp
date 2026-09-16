#include "benchmark.hpp"
#include "intra_node_exchange.hpp"

namespace {

std::unique_ptr<trident::IntraNodeExchange> makeExchange(const trident::ProcessGrid& grid,
                                                       const trident::ExecutionPlan&) {
    return std::make_unique<trident::TwoSidedIntraNodeExchange>(grid);
}

}  // namespace

int main(int argc, char** argv) {
    return trident::runBenchmark(argc, argv, "trident_two_sided", makeExchange);
}
