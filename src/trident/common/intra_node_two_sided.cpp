#include "intra_node_two_sided.hpp"

#include <algorithm>

namespace trident {

void TwoSidedIntraNodeExchange::assemble(const CsrMatrix& slice, const Stage& stage, CsrMatrix& panel) {
    panel.rows = stage.panel.rows;
    panel.cols = stage.panel.cols;
    panel.rowPtr.resize(static_cast<std::size_t>(panel.rows) + 1);
    panel.columnIndices.resize(stage.panel.nnz);
    panel.values.resize(stage.panel.nnz);
    const int local = grid_.nodeRank;
    const int rowOffset = stage.rowOffsets[local];
    const int nnzOffset = stage.nnzOffsets[local];
    std::copy_n(slice.rowPtr.begin(), slice.rows, panel.rowPtr.begin() + rowOffset);
    std::copy(slice.columnIndices.begin(), slice.columnIndices.end(), panel.columnIndices.begin() + nnzOffset);
    std::copy(slice.values.begin(), slice.values.end(), panel.values.begin() + nnzOffset);

    std::vector<MPI_Request> requests;
    requests.reserve(static_cast<std::size_t>(grid_.nodeSize - 1) * 6);
    auto receive = [&](void* buffer, int count, MPI_Datatype type, int peer, int tag) {
        if (count == 0) return;
        requests.push_back(MPI_REQUEST_NULL);
        checkMpi(MPI_Irecv(buffer, count, type, peer, tag, grid_.node, &requests.back()),
                 "MPI_Irecv(intra B)", grid_.world);
    };
    auto send = [&](const void* buffer, int count, MPI_Datatype type, int peer, int tag) {
        if (count == 0) return;
        requests.push_back(MPI_REQUEST_NULL);
        checkMpi(MPI_Isend(buffer, count, type, peer, tag, grid_.node, &requests.back()),
                 "MPI_Isend(intra B)", grid_.world);
    };
    for (int peer = 0; peer < grid_.nodeSize; ++peer) {
        if (peer == local) continue;
        receive(panel.rowPtr.data() + stage.rowOffsets[peer], stage.rowCounts[peer], MPI_INT, peer, 40);
        if (stage.nnzCounts[peer] > 0) {
            receive(panel.columnIndices.data() + stage.nnzOffsets[peer], stage.nnzCounts[peer], MPI_INT, peer, 41);
            receive(panel.values.data() + stage.nnzOffsets[peer], stage.nnzCounts[peer], MPI_DOUBLE, peer, 42);
        }
    }
    for (int peer = 0; peer < grid_.nodeSize; ++peer) {
        if (peer == local) continue;
        send(slice.rowPtr.data(), slice.rows, MPI_INT, peer, 40);
        send(slice.columnIndices.data(), stage.nnzCounts[local], MPI_INT, peer, 41);
        send(slice.values.data(), stage.nnzCounts[local], MPI_DOUBLE, peer, 42);
    }
    waitAll(requests, grid_.world);
    // Slice pointers are local; rebase them after every receive has completed.
    for (int peer = 0; peer < grid_.nodeSize; ++peer) {
        for (int r = 0; r < stage.rowCounts[peer]; ++r) {
            panel.rowPtr[stage.rowOffsets[peer] + r] += stage.nnzOffsets[peer];
        }
    }
    panel.rowPtr.back() = stage.panel.nnz;
}

}  // namespace trident
