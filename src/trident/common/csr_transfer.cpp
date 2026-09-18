#include "csr_transfer.hpp"

namespace trident {

void reserveTile(CsrMatrix& tile, TileShape shape) {
    tile.rowPtr.reserve(static_cast<std::size_t>(shape.rows) + 1);
    tile.columnIndices.reserve(shape.nnz);
    tile.values.reserve(shape.nnz);
}

void resizeTile(CsrMatrix& tile, TileShape shape) {
    tile.rows = shape.rows;
    tile.cols = shape.cols;
    tile.rowPtr.resize(static_cast<std::size_t>(shape.rows) + 1);
    tile.columnIndices.resize(shape.nnz);
    tile.values.resize(shape.nnz);
}

void receiveTile(CsrMatrix& tile, int source, int tag, MPI_Comm comm,
                 std::vector<MPI_Request>& requests) {
    const auto offset = requests.size();
    requests.resize(offset + 3, MPI_REQUEST_NULL);
    checkMpi(MPI_Irecv(tile.rowPtr.data(), mpiCount(tile.rowPtr.size()), MPI_INT, source, tag,
                        comm, &requests[offset]), "MPI_Irecv(inter pointers)", comm);
    checkMpi(MPI_Irecv(tile.columnIndices.data(), mpiCount(tile.values.size()), MPI_INT, source,
                        tag + 1, comm, &requests[offset + 1]), "MPI_Irecv(inter indices)", comm);
    checkMpi(MPI_Irecv(tile.values.data(), mpiCount(tile.values.size()), MPI_DOUBLE, source,
                        tag + 2, comm, &requests[offset + 2]), "MPI_Irecv(inter values)", comm);
}

void sendTile(const CsrMatrix& tile, int target, int tag, MPI_Comm comm,
              std::vector<MPI_Request>& requests) {
    const auto offset = requests.size();
    requests.resize(offset + 3, MPI_REQUEST_NULL);
    checkMpi(MPI_Isend(tile.rowPtr.data(), mpiCount(tile.rowPtr.size()), MPI_INT, target, tag,
                        comm, &requests[offset]), "MPI_Isend(inter pointers)", comm);
    checkMpi(MPI_Isend(tile.columnIndices.data(), mpiCount(tile.values.size()), MPI_INT, target,
                        tag + 1, comm, &requests[offset + 1]), "MPI_Isend(inter indices)", comm);
    checkMpi(MPI_Isend(tile.values.data(), mpiCount(tile.values.size()), MPI_DOUBLE, target,
                        tag + 2, comm, &requests[offset + 2]), "MPI_Isend(inter values)", comm);
}

}  // namespace trident
