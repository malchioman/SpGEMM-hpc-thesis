#pragma once

#include "rma_get.hpp"

namespace trident {

class PipelinedGetInterNodeExchange final : public InterNodeExchange {
public:
    // The immutable plan and exposed input storage must outlive this backend.
    PipelinedGetInterNodeExchange(const ProcessGrid& grid, const ExecutionPlan& plan,
                                 const CsrMatrix& a, const CsrMatrix& b);
    void begin(const CsrMatrix& a, const CsrMatrix& b) override;
    // Stages must be supplied from the bound plan, in order, with one stable workspace.
    void fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
               ProductWorkspace& workspace) override;
    void finish() override;
    void close() override;

private:
    const ExecutionPlan& plan_;
    CsrMatrix incomingA_;
    CsrMatrix incomingB_;
    // Destroy the transport before its destination buffers if a product is interrupted.
    GetTileTransport transfer_;
    std::size_t nextStage_ = 0;
};

}  // namespace trident
