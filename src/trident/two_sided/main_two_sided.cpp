#include "benchmark.hpp"
#include "inter_node_exchange.hpp"
#include "intra_node_two_sided.hpp"

namespace {

std::unique_ptr<trident::IntraNodeExchange> makeIntra(const trident::ProcessGrid& grid,
                                                   const trident::ExecutionPlan&) {
    return std::make_unique<trident::TwoSidedIntraNodeExchange>(grid);
}

std::unique_ptr<trident::InterNodeExchange> makeInter(const trident::ProcessGrid& grid,
                                                   const trident::ExecutionPlan&,
                                                   const CsrMatrix&, const CsrMatrix&,
                                                   trident::ProductWorkspace&) {
    return std::make_unique<trident::TwoSidedInterNodeExchange>(grid);
}

}  // namespace

int main(int argc, char** argv) {
    return trident::runBenchmark(argc, argv, {"trident_two_sided", makeIntra, makeInter});
}
