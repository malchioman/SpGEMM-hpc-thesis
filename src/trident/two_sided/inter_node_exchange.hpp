#pragma once

#include "execution.hpp"

namespace trident {

class TwoSidedInterNodeExchange final : public InterNodeExchange {
public:
    explicit TwoSidedInterNodeExchange(const ProcessGrid& grid) : grid_(grid) {}
    // All transfers complete within fetch; there is no background service to manage.
    void begin(const CsrMatrix&, const CsrMatrix&) override {}
    void fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
               ProductWorkspace& workspace) override;
    void finish() override {}
    void close() override {}

private:
    const ProcessGrid& grid_;
};

}  // namespace trident
