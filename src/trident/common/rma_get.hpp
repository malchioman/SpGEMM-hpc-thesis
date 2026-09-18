#pragma once

#include "execution.hpp"

namespace trident {

// Shared passive-target GET transport. At most one A/B pair is outstanding.
class GetTileTransport {
public:
    GetTileTransport(const ProcessGrid& grid, const CsrMatrix& a, const CsrMatrix& b);
    GetTileTransport(const GetTileTransport&) = delete;
    GetTileTransport& operator=(const GetTileTransport&) = delete;
    ~GetTileTransport();
    void begin(const CsrMatrix& a, const CsrMatrix& b);
    // Destination arrays must remain untouched and alive until completeFetch().
    void startFetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                    CsrMatrix& nextA, CsrMatrix& nextB);
    void completeFetch();
    void finish();
    void close();

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
    int aSource_ = 0;
    int bSource_ = 0;
    bool pending_ = false;
    bool active_ = false;
    bool closed_ = false;
};

}  // namespace trident
