#include "spgemm_exchange.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kRequestCountTag = 200;
constexpr int kRequestRowsTag = 201;
constexpr int kSparseRowNnzTag = 202;
constexpr int kSparseColumnTag = 203;
constexpr int kSparseValueTag = 204;

template <typename T>
T* dataOrNull(std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

template <typename T>
const T* dataOrNull(const std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

template <typename T>
T* dataAtOrNull(std::vector<T>& values, int offset, int count) {
    return count == 0 ? nullptr : values.data() + offset;
}

struct SparseRowView {
    const int* columns = nullptr;
    const double* values = nullptr;
    int nonZeros = 0;
};

SparseRowView localSparseRow(const CsrMatrix& matrix, int localRow) {
    const int first = matrix.rowPtr[localRow];
    const int last = matrix.rowPtr[localRow + 1];
    const int nonZeros = last - first;
    return {nonZeros == 0 ? nullptr : matrix.columnIndices.data() + first,
            nonZeros == 0 ? nullptr : matrix.values.data() + first, nonZeros};
}

SparseRowView remoteSparseRow(const RemoteSparseRows& rows, int slot) {
    const int first = rows.rowPtr[slot];
    const int last = rows.rowPtr[slot + 1];
    const int nonZeros = last - first;
    return {nonZeros == 0 ? nullptr : rows.columnIndices.data() + first,
            nonZeros == 0 ? nullptr : rows.values.data() + first, nonZeros};
}

int sumCounts(const std::vector<int>& counts) {
    return std::accumulate(counts.begin(), counts.end(), 0);
}

void packRequestedRows(const CsrMatrix& localMatrixB, RowBlock localBBlock,
                       const std::vector<int>& requests, std::vector<int>& rowNonZeros,
                       std::vector<int>& columns, std::vector<double>& values,
                       MPI_Comm communicator) {
    rowNonZeros.resize(requests.size());
    int totalNonZeros = 0;
    for (int index = 0; index < static_cast<int>(requests.size()); ++index) {
        const int globalRow = requests[index];
        if (globalRow < localBBlock.firstRow || globalRow >= localBBlock.firstRow + localBBlock.rows) {
            fail("received a request for a non-local sparse row of B", communicator);
        }
        const int localRow = globalRow - localBBlock.firstRow;
        const int nonZeros = localMatrixB.rowPtr[localRow + 1] - localMatrixB.rowPtr[localRow];
        rowNonZeros[index] = nonZeros;
        totalNonZeros += nonZeros;
    }

    columns.clear();
    values.clear();
    columns.reserve(totalNonZeros);
    values.reserve(totalNonZeros);
    for (const int globalRow : requests) {
        const int localRow = globalRow - localBBlock.firstRow;
        const int first = localMatrixB.rowPtr[localRow];
        const int last = localMatrixB.rowPtr[localRow + 1];
        columns.insert(columns.end(), localMatrixB.columnIndices.begin() + first,
                       localMatrixB.columnIndices.begin() + last);
        values.insert(values.end(), localMatrixB.values.begin() + first,
                      localMatrixB.values.begin() + last);
    }
}

SparseRowView bRowForColumn(const CsrMatrix& localMatrixB, RowBlock localBBlock,
                            const RemoteSparseRows& remoteBRows, int globalBRow,
                            MPI_Comm communicator) {
    if (globalBRow >= localBBlock.firstRow && globalBRow < localBBlock.firstRow + localBBlock.rows) {
        return localSparseRow(localMatrixB, globalBRow - localBBlock.firstRow);
    }

    const auto slot = remoteBRows.plan.rowToSlot.find(globalBRow);
    if (slot == remoteBRows.plan.rowToSlot.end()) {
        fail("missing a remote sparse row required by A", communicator);
    }
    return remoteSparseRow(remoteBRows, slot->second);
}

}  // namespace

RemoteRowPlan buildRemoteRowPlan(const CsrMatrix& localMatrix, const std::vector<RowBlock>& blocks,
                                 int rank, MPI_Comm communicator) {
    RemoteRowPlan plan;
    plan.rowsByPeer.resize(blocks.size());

    for (const int column : localMatrix.columnIndices) {
        const int owner = ownerOfRow(column, blocks);
        if (owner < 0) {
            fail("sparse matrix A contains a column index outside B's row range", communicator);
        }
        if (owner != rank) {
            plan.rowsByPeer[owner].push_back(column);
        }
    }

    plan.peerOffsets.resize(blocks.size() + 1, 0);
    int totalRemoteRows = 0;
    for (int peer = 0; peer < static_cast<int>(blocks.size()); ++peer) {
        auto& rows = plan.rowsByPeer[peer];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        plan.peerOffsets[peer] = totalRemoteRows;
        totalRemoteRows += static_cast<int>(rows.size());
    }
    plan.peerOffsets.back() = totalRemoteRows;

    plan.flatRows.reserve(totalRemoteRows);
    plan.rowToSlot.reserve(totalRemoteRows);
    int slot = 0;
    for (const auto& rows : plan.rowsByPeer) {
        for (const int row : rows) {
            plan.flatRows.push_back(row);
            plan.rowToSlot.emplace(row, slot);
            ++slot;
        }
    }
    return plan;
}

RemoteSparseRows makeRemoteSparseRows(RemoteRowPlan plan) {
    RemoteSparseRows remoteRows;
    remoteRows.plan = std::move(plan);
    remoteRows.rowPtr.assign(remoteRows.plan.flatRows.size() + 1, 0);
    return remoteRows;
}

RemoteSparseRows exchangeRemoteSparseRowsTwoSided(const CsrMatrix& localMatrixA,
                                                  const CsrMatrix& localMatrixB,
                                                  const std::vector<RowBlock>& bBlocks,
                                                  RowBlock localBBlock, int rank, int ranks,
                                                  MPI_Comm communicator) {
    RemoteSparseRows remoteRows =
        makeRemoteSparseRows(buildRemoteRowPlan(localMatrixA, bBlocks, rank, communicator));

    std::vector<int> incomingCounts(ranks, 0);
    std::vector<int> outgoingCounts(ranks, 0);
    for (int peer = 0; peer < ranks; ++peer) {
        outgoingCounts[peer] = static_cast<int>(remoteRows.plan.rowsByPeer[peer].size());
    }

    std::vector<MPI_Request> requests;
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(&incomingCounts[peer], 1, MPI_INT, peer, kRequestCountTag, communicator,
                           &request),
                 "MPI_Irecv(request count)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(&outgoingCounts[peer], 1, MPI_INT, peer, kRequestCountTag, communicator,
                           &request),
                 "MPI_Isend(request count)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<int>> incomingRequests(ranks);
    requests.clear();
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        incomingRequests[peer].resize(incomingCounts[peer]);
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(dataOrNull(incomingRequests[peer]), incomingCounts[peer], MPI_INT, peer,
                           kRequestRowsTag, communicator, &request),
                 "MPI_Irecv(row requests)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(dataOrNull(remoteRows.plan.rowsByPeer[peer]),
                           static_cast<int>(remoteRows.plan.rowsByPeer[peer].size()), MPI_INT, peer,
                           kRequestRowsTag, communicator, &request),
                 "MPI_Isend(row requests)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<int>> incomingRowNonZeros(ranks);
    std::vector<std::vector<int>> outgoingRowNonZeros(ranks);
    std::vector<std::vector<int>> outgoingColumns(ranks);
    std::vector<std::vector<double>> outgoingValues(ranks);
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        incomingRowNonZeros[peer].resize(outgoingCounts[peer]);
        packRequestedRows(localMatrixB, localBBlock, incomingRequests[peer],
                          outgoingRowNonZeros[peer], outgoingColumns[peer], outgoingValues[peer],
                          communicator);
    }

    requests.clear();
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(dataOrNull(incomingRowNonZeros[peer]), outgoingCounts[peer], MPI_INT,
                           peer, kSparseRowNnzTag, communicator, &request),
                 "MPI_Irecv(sparse row nnz)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(dataOrNull(outgoingRowNonZeros[peer]),
                           static_cast<int>(outgoingRowNonZeros[peer].size()), MPI_INT, peer,
                           kSparseRowNnzTag, communicator, &request),
                 "MPI_Isend(sparse row nnz)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    int totalRemoteNonZeros = 0;
    for (int peer = 0; peer < ranks; ++peer) {
        const int peerOffset = remoteRows.plan.peerOffsets[peer];
        for (int index = 0; index < static_cast<int>(incomingRowNonZeros[peer].size()); ++index) {
            remoteRows.rowPtr[peerOffset + index] = totalRemoteNonZeros;
            totalRemoteNonZeros += incomingRowNonZeros[peer][index];
            remoteRows.rowPtr[peerOffset + index + 1] = totalRemoteNonZeros;
        }
    }
    remoteRows.columnIndices.resize(totalRemoteNonZeros);
    remoteRows.values.resize(totalRemoteNonZeros);

    requests.clear();
    requests.reserve(4 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        const int receiveOffset = remoteRows.rowPtr[remoteRows.plan.peerOffsets[peer]];
        const int receiveCount = sumCounts(incomingRowNonZeros[peer]);
        const int sendCount = static_cast<int>(outgoingColumns[peer].size());
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(dataAtOrNull(remoteRows.columnIndices, receiveOffset, receiveCount),
                           receiveCount, MPI_INT, peer, kSparseColumnTag, communicator, &request),
                 "MPI_Irecv(sparse row columns)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Irecv(dataAtOrNull(remoteRows.values, receiveOffset, receiveCount), receiveCount,
                           MPI_DOUBLE, peer, kSparseValueTag, communicator, &request),
                 "MPI_Irecv(sparse row values)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(dataOrNull(outgoingColumns[peer]), sendCount, MPI_INT, peer,
                           kSparseColumnTag, communicator, &request),
                 "MPI_Isend(sparse row columns)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(dataOrNull(outgoingValues[peer]), sendCount, MPI_DOUBLE, peer,
                           kSparseValueTag, communicator, &request),
                 "MPI_Isend(sparse row values)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    return remoteRows;
}

CsrMatrix spgemm(const CsrMatrix& localMatrixA, RowBlock localBBlock,
                 const CsrMatrix& localMatrixB, const RemoteSparseRows& remoteBRows,
                 MPI_Comm communicator) {
    CsrMatrix result;
    result.rows = localMatrixA.rows;
    result.cols = localMatrixB.cols;
    result.rowPtr.resize(localMatrixA.rows + 1, 0);

    std::vector<std::vector<int>> columnsByRow(localMatrixA.rows);
    std::vector<std::vector<double>> valuesByRow(localMatrixA.rows);

#pragma omp parallel
    {
        std::unordered_map<int, double> accumulator;
        std::vector<int> touchedColumns;

#pragma omp for schedule(runtime)
        for (int row = 0; row < localMatrixA.rows; ++row) {
            accumulator.clear();
            touchedColumns.clear();

            for (int aEntry = localMatrixA.rowPtr[row]; aEntry < localMatrixA.rowPtr[row + 1];
                 ++aEntry) {
                const int bRow = localMatrixA.columnIndices[aEntry];
                const double aValue = localMatrixA.values[aEntry];
                const SparseRowView bValues =
                    bRowForColumn(localMatrixB, localBBlock, remoteBRows, bRow, communicator);
                for (int bEntry = 0; bEntry < bValues.nonZeros; ++bEntry) {
                    const int column = bValues.columns[bEntry];
                    const double product = aValue * bValues.values[bEntry];
                    const auto [position, inserted] = accumulator.emplace(column, product);
                    if (inserted) {
                        touchedColumns.push_back(column);
                    } else {
                        position->second += product;
                    }
                }
            }

            std::sort(touchedColumns.begin(), touchedColumns.end());
            auto& rowColumns = columnsByRow[row];
            auto& rowValues = valuesByRow[row];
            rowColumns.reserve(touchedColumns.size());
            rowValues.reserve(touchedColumns.size());
            for (const int column : touchedColumns) {
                const double value = accumulator.at(column);
                if (value != 0.0) {
                    rowColumns.push_back(column);
                    rowValues.push_back(value);
                }
            }
        }
    }

    int totalNonZeros = 0;
    for (int row = 0; row < result.rows; ++row) {
        totalNonZeros += static_cast<int>(valuesByRow[row].size());
        result.rowPtr[row + 1] = totalNonZeros;
    }
    result.columnIndices.reserve(totalNonZeros);
    result.values.reserve(totalNonZeros);
    for (int row = 0; row < result.rows; ++row) {
        result.columnIndices.insert(result.columnIndices.end(), columnsByRow[row].begin(),
                                    columnsByRow[row].end());
        result.values.insert(result.values.end(), valuesByRow[row].begin(), valuesByRow[row].end());
    }
    return result;
}
