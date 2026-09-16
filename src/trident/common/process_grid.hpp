#pragma once

#include "spgemm_common.hpp"

namespace trident {

// CPU topology for the (i,j,k) coordinates used by Trident; see docs/trident.md.
class ProcessGrid {
public:
    explicit ProcessGrid(MPI_Comm communicator, int logicalNodeSize = 0);
    ProcessGrid(const ProcessGrid&) = delete;
    ProcessGrid& operator=(const ProcessGrid&) = delete;
    // Explicit collective cleanup: exceptions must reach MPI_Abort without unwinding collectives.
    void close();
    int rankAt(int i, int j, int k) const;

    MPI_Comm world = MPI_COMM_NULL;
    MPI_Comm node = MPI_COMM_NULL;
    int rank = 0;
    int size = 0;
    int nodeRank = 0;
    int nodeSize = 0;
    int nodeCount = 0;
    int side = 0;
    int row = 0;
    int col = 0;
    bool emulated = false;
    std::vector<int> coordinateRanks;
};

}  // namespace trident
