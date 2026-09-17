#include "inter_node_exchange.hpp"

#include "csr_transfer.hpp"

namespace trident {

void TwoSidedInterNodeExchange::fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                                    ProductWorkspace& workspace) {
    resizeTile(workspace.a, stage.a);
    resizeTile(workspace.b, stage.b);
    std::vector<MPI_Request> requests;
    requests.reserve(12);
    if (stage.aSource == grid_.rank) {
        workspace.a = a;
    } else {
        receiveTile(workspace.a, stage.aSource, 20, grid_.world, requests);
    }
    if (stage.bSource == grid_.rank) {
        workspace.b = b;
    } else {
        receiveTile(workspace.b, stage.bSource, 30, grid_.world, requests);
    }
    // Invert h=(i+j+round)%q: each static owner has exactly one consumer per round.
    if (stage.aTarget != grid_.rank) {
        sendTile(a, stage.aTarget, 20, grid_.world, requests);
    }
    if (stage.bTarget != grid_.rank) {
        sendTile(b, stage.bTarget, 30, grid_.world, requests);
    }
    waitAll(requests, grid_.world);
}

}  // namespace trident
