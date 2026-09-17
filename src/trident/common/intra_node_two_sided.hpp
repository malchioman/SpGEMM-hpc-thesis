#pragma once

#include "execution.hpp"

namespace trident {

// Node-local B aggregation shared by the two-sided and hybrid executables.
class TwoSidedIntraNodeExchange final : public IntraNodeExchange {
public:
    explicit TwoSidedIntraNodeExchange(const ProcessGrid& grid) : grid_(grid) {}
    void assemble(const CsrMatrix& slice, const Stage& stage, CsrMatrix& panel) override;

private:
    const ProcessGrid& grid_;
};

}  // namespace trident
