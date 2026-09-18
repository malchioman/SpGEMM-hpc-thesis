#pragma once

#include "rma_get.hpp"

namespace trident {

class GetInterNodeExchange final : public InterNodeExchange {
public:
    // Input arrays must retain their addresses and sizes until close().
    GetInterNodeExchange(const ProcessGrid& grid, const CsrMatrix& a, const CsrMatrix& b);
    GetInterNodeExchange(const GetInterNodeExchange&) = delete;
    GetInterNodeExchange& operator=(const GetInterNodeExchange&) = delete;
    void begin(const CsrMatrix& a, const CsrMatrix& b) override;
    void fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
               ProductWorkspace& workspace) override;
    void finish() override;
    void close() override;

private:
    GetTileTransport transfer_;
};

}  // namespace trident
