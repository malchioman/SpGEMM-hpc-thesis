#pragma once

#include "spmm_common.hpp"

#include <unordered_map>
#include <vector>

struct RemoteRowPlan {
    std::vector<std::vector<int>> rowsByPeer;
    std::vector<int> peerOffsets;
    std::vector<int> flatRows;
    std::unordered_map<int, int> rowToSlot;
};

struct RemoteDenseRows {
    RemoteRowPlan plan;
    std::vector<double> values;
};

RemoteRowPlan buildRemoteRowPlan(const CsrMatrix& localMatrix, const std::vector<RowBlock>& blocks,
                                 int rank, MPI_Comm communicator);
RemoteDenseRows makeRemoteDenseRows(RemoteRowPlan plan, int denseCols);
RemoteDenseRows exchangeRemoteDenseRowsTwoSided(const CsrMatrix& localMatrix,
                                                const std::vector<RowBlock>& blocks,
                                                RowBlock localBlock,
                                                const std::vector<double>& ownedDenseRows,
                                                int denseCols, int rank, int ranks,
                                                MPI_Comm communicator);
void spmm(const CsrMatrix& matrix, RowBlock localBlock, const std::vector<double>& ownedDenseRows,
          const RemoteDenseRows& remoteDenseRows, int denseCols, std::vector<double>& result);
