#include "benchmark.hpp"
#include "inter_node_exchange.hpp"
#include "intra_node_two_sided.hpp"

namespace {

std::unique_ptr<trident::IntraNodeExchange> makeIntra(const trident::ProcessGrid& grid,
                                                   const trident::ExecutionPlan&) {
    return std::make_unique<trident::TwoSidedIntraNodeExchange>(grid);
}

std::unique_ptr<trident::InterNodeExchange> makeInter(const trident::ProcessGrid& grid,
                                                   const trident::ExecutionPlan& plan,
                                                   const CsrMatrix& a, const CsrMatrix& b) {
    return std::make_unique<trident::PipelinedGetInterNodeExchange>(grid, plan, a, b);
}

}  // namespace

int main(int argc, char** argv) {
    return trident::runBenchmark(argc, argv, {"trident_get_pipeline", makeIntra, makeInter,
        MPI_THREAD_FUNNELED, "trident_get_pipeline_csr_v1", "one_sided_get_pipeline"});
}
