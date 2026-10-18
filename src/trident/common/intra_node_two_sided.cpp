#include "intra_node_two_sided.hpp"

#include <algorithm>

namespace trident {

void TwoSidedIntraNodeExchange::assemble(const CsrMatrix& slice, const Stage& stage, CsrMatrix& panel) {
    panel.rows = stage.panel.rows;
    panel.cols = stage.panel.cols;
    panel.rowPtr.resize(static_cast<std::size_t>(panel.rows) + 1);
    panel.columnIndices.resize(csrSize<int>(stage.panel.nnz));
    panel.values.resize(csrSize<double>(stage.panel.nnz));
    const int local = grid_.nodeRank;
    const int rowOffset = stage.rowOffsets[local];
    const CsrOffset nnzOffset = stage.nnzOffsets[local];
    std::copy_n(slice.rowPtr.begin(), slice.rows, panel.rowPtr.begin() + rowOffset);
    std::copy(slice.columnIndices.begin(), slice.columnIndices.end(), panel.columnIndices.begin() + nnzOffset);
    std::copy(slice.values.begin(), slice.values.end(), panel.values.begin() + nnzOffset);

    std::vector<MPI_Request> requests;
    requests.reserve(static_cast<std::size_t>(grid_.nodeSize - 1) * 6);
    auto receive = [&](auto* buffer, CsrOffset count, int peer, int tag) {
        csrIrecv(buffer, count, peer, tag, grid_.node, requests);
    };
    auto send = [&](const auto* buffer, CsrOffset count, int peer, int tag) {
        csrIsend(buffer, count, peer, tag, grid_.node, requests);
    };
    for (int peer = 0; peer < grid_.nodeSize; ++peer) {
        if (peer == local) continue;
        receive(panel.rowPtr.data() + stage.rowOffsets[peer], stage.rowCounts[peer], peer, 40);
        if (stage.nnzCounts[peer] > 0) {
            receive(panel.columnIndices.data() + stage.nnzOffsets[peer], stage.nnzCounts[peer], peer, 41);
            receive(panel.values.data() + stage.nnzOffsets[peer], stage.nnzCounts[peer], peer, 42);
        }
    }
    for (int peer = 0; peer < grid_.nodeSize; ++peer) {
        if (peer == local) continue;
        send(slice.rowPtr.data(), slice.rows, peer, 40);
        send(slice.columnIndices.data(), stage.nnzCounts[local], peer, 41);
        send(slice.values.data(), stage.nnzCounts[local], peer, 42);
    }
    waitAll(requests, grid_.world);
    // Slice pointers are local; rebase them after every receive has completed.
    for (int peer = 0; peer < grid_.nodeSize; ++peer) {
        for (int r = 0; r < stage.rowCounts[peer]; ++r) {
            auto& pointer = panel.rowPtr[stage.rowOffsets[peer] + r];
            pointer = checkedCsrAdd(pointer, stage.nnzOffsets[peer], "panel row pointer");
        }
    }
    panel.rowPtr.back() = stage.panel.nnz;
}

}  // namespace trident
