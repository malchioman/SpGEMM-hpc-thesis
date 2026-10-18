#include "csr_transfer.hpp"

namespace trident {

void reserveTile(CsrMatrix& tile, TileShape shape) {
    tile.rowPtr.reserve(static_cast<std::size_t>(shape.rows) + 1);
    tile.columnIndices.reserve(csrSize<int>(shape.nnz));
    tile.values.reserve(csrSize<double>(shape.nnz));
}

void resizeTile(CsrMatrix& tile, TileShape shape) {
    tile.rows = shape.rows;
    tile.cols = shape.cols;
    tile.rowPtr.resize(static_cast<std::size_t>(shape.rows) + 1);
    tile.columnIndices.resize(csrSize<int>(shape.nnz));
    tile.values.resize(csrSize<double>(shape.nnz));
}

void receiveTile(CsrMatrix& tile, int source, int tag, MPI_Comm comm,
                 std::vector<MPI_Request>& requests) {
    csrIrecv(tile.rowPtr.data(), checkedCsrCount(tile.rowPtr.size()), source, tag, comm, requests);
    csrIrecv(tile.columnIndices.data(), checkedCsrCount(tile.values.size()), source, tag + 1, comm, requests);
    csrIrecv(tile.values.data(), checkedCsrCount(tile.values.size()), source, tag + 2, comm, requests);
}

void sendTile(const CsrMatrix& tile, int target, int tag, MPI_Comm comm,
              std::vector<MPI_Request>& requests) {
    csrIsend(tile.rowPtr.data(), checkedCsrCount(tile.rowPtr.size()), target, tag, comm, requests);
    csrIsend(tile.columnIndices.data(), checkedCsrCount(tile.values.size()), target, tag + 1, comm, requests);
    csrIsend(tile.values.data(), checkedCsrCount(tile.values.size()), target, tag + 2, comm, requests);
}

}  // namespace trident
