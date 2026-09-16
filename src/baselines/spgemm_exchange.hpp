#pragma once

#include "spgemm_common.hpp"

#include <unordered_map>
#include <vector>

struct RemoteRowPlan {
    std::vector<std::vector<int>> rowsByPeer;
    std::vector<int> peerOffsets;
    std::vector<int> flatRows;
    std::unordered_map<int, int> rowToSlot;
};

struct RemoteSparseRows {
    RemoteRowPlan plan;
    std::vector<int> rowPtr;
    std::vector<int> columnIndices;
    std::vector<double> values;
};

RemoteRowPlan buildRemoteRowPlan(const CsrMatrix& localMatrix, const std::vector<RowBlock>& blocks,
                                 int rank, MPI_Comm communicator);
RemoteSparseRows makeRemoteSparseRows(RemoteRowPlan plan);
struct TwoSidedPlan {
    RemoteSparseRows remoteRows;
    std::vector<std::vector<int>> incomingRequests;
    std::vector<std::vector<int>> outgoingColumns;
    std::vector<std::vector<double>> outgoingValues;
};

TwoSidedPlan buildTwoSidedPlan(const CsrMatrix& localMatrixA, const CsrMatrix& localMatrixB,
                              const std::vector<RowBlock>& bBlocks, RowBlock localBBlock,
                              int rank, int ranks, MPI_Comm communicator);
void exchangeRemoteSparseRowsTwoSided(const CsrMatrix& localMatrixB, RowBlock localBBlock,
                                     TwoSidedPlan& plan, int rank, MPI_Comm communicator);
CsrMatrix spgemm(const CsrMatrix& localMatrixA, RowBlock localBBlock,
                 const CsrMatrix& localMatrixB, const RemoteSparseRows& remoteBRows,
                 MPI_Comm communicator);
