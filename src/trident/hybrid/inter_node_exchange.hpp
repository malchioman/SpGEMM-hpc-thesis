#pragma once

#include "execution.hpp"

#include <cstdint>
#include <thread>

namespace trident {

// Bounded request queues: one generation slot per consumer and input matrix.
class HybridInterNodeExchange final : public InterNodeExchange {
public:
    explicit HybridInterNodeExchange(const ProcessGrid& grid);
    ~HybridInterNodeExchange() override;
    void begin(const CsrMatrix& a, const CsrMatrix& b) override;
    void fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
               ProductWorkspace& workspace) override;
    void finish() override;
    void close() override;
    std::uint64_t publishedRequests() const { return published_; }
    std::uint64_t servedRequests() const { return served_; }

private:
    void notify(int target, int slot);
    void serve(const CsrMatrix& a, const CsrMatrix& b, std::uint64_t generation);
    const ProcessGrid& grid_;
    MPI_Comm payload_ = MPI_COMM_NULL;
    MPI_Win requests_ = MPI_WIN_NULL;
    std::thread service_;
    std::uint64_t generation_ = 0;
    std::uint64_t published_ = 0;
    std::uint64_t served_ = 0;
    bool active_ = false;
};

}  // namespace trident
