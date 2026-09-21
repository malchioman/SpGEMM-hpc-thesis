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
                                                   const CsrMatrix& a, const CsrMatrix& b,
                                                   trident::ProductWorkspace& workspace) {
    return std::make_unique<trident::PutInterNodeExchange>(grid, plan, a, b, workspace);
}

}  // namespace

int main(int argc, char** argv) {
    return trident::runBenchmark(argc, argv, {"trident_put", makeIntra, makeInter,
        MPI_THREAD_FUNNELED, "trident_put_staged_csr_v1", "one_sided_put"});
}
