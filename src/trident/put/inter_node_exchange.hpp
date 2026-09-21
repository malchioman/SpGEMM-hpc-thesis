#pragma once

#include "execution.hpp"

namespace trident {

class PutInterNodeExchange final : public InterNodeExchange {
public:
    // The immutable plan and workspace storage must outlive collective close().
    PutInterNodeExchange(const ProcessGrid& grid, const ExecutionPlan& plan,
                         const CsrMatrix& a, const CsrMatrix& b, ProductWorkspace& workspace);
    PutInterNodeExchange(const PutInterNodeExchange&) = delete;
    PutInterNodeExchange& operator=(const PutInterNodeExchange&) = delete;
    ~PutInterNodeExchange() override;
    void begin(const CsrMatrix& a, const CsrMatrix& b) override;
    void fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
               ProductWorkspace& workspace) override;
    void finish() override;
    void close() override;

private:
    struct ReceiveWindow {
        TileShape maximum;
        std::array<const void*, 3> bases;
        std::array<MPI_Win, 3> windows{MPI_WIN_NULL, MPI_WIN_NULL, MPI_WIN_NULL};
    };
    ReceiveWindow expose(CsrMatrix& tile, bool operandA);
    void checkWorkspace() const;
    void checkInputs(const CsrMatrix& a, const CsrMatrix& b) const;
    void syncReceivers();
    void putTile(const CsrMatrix& tile, int target, const ReceiveWindow& receiver);
    void complete(int target, const ReceiveWindow& receiver);
    const ProcessGrid& grid_;
    const ExecutionPlan& plan_;
    ProductWorkspace& workspace_;
    TileShape aShape_;
    TileShape bShape_;
    ReceiveWindow aWindow_;
    ReceiveWindow bWindow_;
    std::size_t nextStage_ = 0;
    bool active_ = false;
    bool closed_ = false;
};

}  // namespace trident
