#include "inter_node_exchange.hpp"

namespace trident {

GetInterNodeExchange::GetInterNodeExchange(const ProcessGrid& grid, const CsrMatrix& a,
                                         const CsrMatrix& b) : transfer_(grid, a, b) {}

void GetInterNodeExchange::begin(const CsrMatrix& a, const CsrMatrix& b) {
    transfer_.begin(a, b);
}

void GetInterNodeExchange::fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                                ProductWorkspace& workspace) {
    transfer_.startFetch(a, b, stage, workspace.a, workspace.b);
    transfer_.completeFetch();
}

void GetInterNodeExchange::finish() { transfer_.finish(); }

void GetInterNodeExchange::close() { transfer_.close(); }

}  // namespace trident
