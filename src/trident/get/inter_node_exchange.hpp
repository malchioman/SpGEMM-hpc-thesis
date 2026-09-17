#pragma once

#include "execution.hpp"

namespace trident {

class GetInterNodeExchange final : public InterNodeExchange {
public:
    // Input arrays must retain their addresses and sizes until close().
    GetInterNodeExchange(const ProcessGrid& grid, const CsrMatrix& a, const CsrMatrix& b);
    GetInterNodeExchange(const GetInterNodeExchange&) = delete;
    GetInterNodeExchange& operator=(const GetInterNodeExchange&) = delete;
    ~GetInterNodeExchange() override;
    void begin(const CsrMatrix& a, const CsrMatrix& b) override;
    void fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
               ProductWorkspace& workspace) override;
    void finish() override;
    void close() override;

private:
    struct InputWindow {
        TileShape shape;
        std::array<const void*, 3> bases;
        std::array<MPI_Win, 3> windows{MPI_WIN_NULL, MPI_WIN_NULL, MPI_WIN_NULL};
    };
    InputWindow expose(const CsrMatrix& input);
    void checkInput(const CsrMatrix& input, const InputWindow& exposed) const;
    void syncInputs();
    void getTile(CsrMatrix& tile, int source, const InputWindow& exposed);
    void complete(int source, const InputWindow& exposed);
    const ProcessGrid& grid_;
    InputWindow a_;
    InputWindow b_;
    bool active_ = false;
    bool closed_ = false;
};

}  // namespace trident
