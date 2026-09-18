#include "inter_node_exchange.hpp"

#include "csr_transfer.hpp"

#include <stdexcept>
#include <utility>

namespace trident {

PipelinedGetInterNodeExchange::PipelinedGetInterNodeExchange(
    const ProcessGrid& grid, const ExecutionPlan& plan, const CsrMatrix& a, const CsrMatrix& b)
    : plan_(plan), transfer_(grid, a, b) {
    for (const auto& stage : plan_.stages) {
        reserveTile(incomingA_, stage.a);
        reserveTile(incomingB_, stage.b);
    }
}

void PipelinedGetInterNodeExchange::begin(const CsrMatrix& a, const CsrMatrix& b) {
    transfer_.begin(a, b);
    nextStage_ = 0;
    if (!plan_.stages.empty()) {
        transfer_.startFetch(a, b, plan_.stages.front(), incomingA_, incomingB_);
    }
}

void PipelinedGetInterNodeExchange::fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                                         ProductWorkspace& workspace) {
    if (nextStage_ >= plan_.stages.size() || &stage != &plan_.stages[nextStage_]) {
        throw std::logic_error("GET pipeline requires the next stage of its bound plan");
    }
    transfer_.completeFetch();
    // Only completed buffers are swapped; the other pair can now receive the next stage.
    std::swap(workspace.a, incomingA_);
    std::swap(workspace.b, incomingB_);
    ++nextStage_;
    if (nextStage_ < plan_.stages.size()) {
        transfer_.startFetch(a, b, plan_.stages[nextStage_], incomingA_, incomingB_);
    }
}

void PipelinedGetInterNodeExchange::finish() {
    if (nextStage_ != plan_.stages.size()) {
        throw std::logic_error("GET pipeline finish before consuming all stages");
    }
    transfer_.finish();
}

void PipelinedGetInterNodeExchange::close() { transfer_.close(); }

}  // namespace trident
