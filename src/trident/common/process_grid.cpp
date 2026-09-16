#include "process_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trident {

ProcessGrid::ProcessGrid(MPI_Comm communicator, int logicalNodeSize) {
    checkMpi(MPI_Comm_dup(communicator, &world), "MPI_Comm_dup", communicator);
    checkMpi(MPI_Comm_rank(world, &rank), "MPI_Comm_rank", world);
    checkMpi(MPI_Comm_size(world, &size), "MPI_Comm_size", world);
    MPI_Comm shared = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared),
             "MPI_Comm_split_type", world);
    int sharedSize = 0;
    checkMpi(MPI_Comm_size(shared, &sharedSize), "MPI_Comm_size(shared)", world);
    if (logicalNodeSize != 0) {
        if (logicalNodeSize < 1 || size % logicalNodeSize != 0 || sharedSize != size) {
            throw std::invalid_argument(
                "--logical-node-size requires a positive divisor of ranks and a single physical node");
        }
        emulated = true;
        checkMpi(MPI_Comm_split(world, rank / logicalNodeSize, rank, &node),
                 "MPI_Comm_split(logical node)", world);
    } else {
        checkMpi(MPI_Comm_dup(shared, &node), "MPI_Comm_dup(node)", world);
    }
    checkMpi(MPI_Comm_free(&shared), "MPI_Comm_free(shared)", world);
    checkMpi(MPI_Comm_rank(node, &nodeRank), "MPI_Comm_rank(node)", world);
    checkMpi(MPI_Comm_size(node, &nodeSize), "MPI_Comm_size(node)", world);
    int minSize = 0;
    int maxSize = 0;
    checkMpi(MPI_Allreduce(&nodeSize, &minSize, 1, MPI_INT, MPI_MIN, world),
             "MPI_Allreduce(min node size)", world);
    checkMpi(MPI_Allreduce(&nodeSize, &maxSize, 1, MPI_INT, MPI_MAX, world),
             "MPI_Allreduce(max node size)", world);
    if (minSize != maxSize) {
        throw std::invalid_argument("Trident requires the same number of MPI ranks on every node");
    }
    nodeCount = size / nodeSize;
    side = static_cast<int>(std::sqrt(nodeCount));
    if (side * side != nodeCount) {
        throw std::invalid_argument("Trident requires a square number of nodes (1, 4, 9, ...)");
    }
    int leader = rank;
    checkMpi(MPI_Bcast(&leader, 1, MPI_INT, 0, node), "MPI_Bcast(node leader)", world);
    std::vector<int> leaders(size);
    std::vector<int> localRanks(size);
    checkMpi(MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, world),
             "MPI_Allgather(node leaders)", world);
    checkMpi(MPI_Allgather(&nodeRank, 1, MPI_INT, localRanks.data(), 1, MPI_INT, world),
             "MPI_Allgather(node ranks)", world);
    auto uniqueLeaders = leaders;
    std::sort(uniqueLeaders.begin(), uniqueLeaders.end());
    uniqueLeaders.erase(std::unique(uniqueLeaders.begin(), uniqueLeaders.end()), uniqueLeaders.end());
    coordinateRanks.resize(size);
    for (int p = 0; p < size; ++p) {
        const int nodeId = static_cast<int>(std::lower_bound(uniqueLeaders.begin(), uniqueLeaders.end(),
                                                            leaders[p]) - uniqueLeaders.begin());
        coordinateRanks[nodeId * nodeSize + localRanks[p]] = p;
        if (p == rank) {
            row = nodeId / side;
            col = nodeId % side;
        }
    }
}

int ProcessGrid::rankAt(int i, int j, int k) const {
    return coordinateRanks.at((i * side + j) * nodeSize + k);
}

void ProcessGrid::close() {
    checkMpi(MPI_Comm_free(&node), "MPI_Comm_free(node)", world);
    checkMpi(MPI_Comm_free(&world), "MPI_Comm_free(world)", MPI_COMM_WORLD);
}

}  // namespace trident
