#pragma once

#include "execution.hpp"

namespace trident {

void resizeTile(CsrMatrix& tile, TileShape shape);
void receiveTile(CsrMatrix& tile, int source, int tag, MPI_Comm comm,
                 std::vector<MPI_Request>& requests);
void sendTile(const CsrMatrix& tile, int target, int tag, MPI_Comm comm,
              std::vector<MPI_Request>& requests);

}  // namespace trident
