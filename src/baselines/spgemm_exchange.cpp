#include "spgemm_exchange.hpp"

#include <algorithm>
#include <cmath>
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
T* dataAtOrNull(std::vector<T>& values, CsrOffset offset, CsrOffset count) {
    return count == 0 ? nullptr : values.data() + offset;
}

struct SparseRowView {
    const int* columns = nullptr;
    const double* values = nullptr;
    CsrOffset nonZeros = 0;
};

SparseRowView localSparseRow(const CsrMatrix& matrix, int localRow) {
    const CsrOffset first = matrix.rowPtr[localRow];
    const CsrOffset last = matrix.rowPtr[localRow + 1];
    const CsrOffset nonZeros = last - first;
    return {nonZeros == 0 ? nullptr : matrix.columnIndices.data() + first,
            nonZeros == 0 ? nullptr : matrix.values.data() + first, nonZeros};
}

SparseRowView remoteSparseRow(const RemoteSparseRows& rows, int slot) {
    const CsrOffset first = rows.rowPtr[slot];
    const CsrOffset last = rows.rowPtr[slot + 1];
    const CsrOffset nonZeros = last - first;
    return {nonZeros == 0 ? nullptr : rows.columnIndices.data() + first,
            nonZeros == 0 ? nullptr : rows.values.data() + first, nonZeros};
}

void prepareRequestedRows(const CsrMatrix& localMatrixB, RowBlock localBBlock,
                       const std::vector<int>& requests, std::vector<CsrOffset>& rowNonZeros,
                       std::vector<int>& columns, std::vector<double>& values,
                       MPI_Comm communicator) {
    rowNonZeros.resize(requests.size());
    CsrOffset totalNonZeros = 0;
    for (int index = 0; index < checkedMatrixDimension(checkedCsrCount(requests.size())); ++index) {
        const int globalRow = requests[index];
        if (globalRow < localBBlock.firstRow || globalRow >= localBBlock.firstRow + localBBlock.rows) {
            fail("received a request for a non-local sparse row of B", communicator);
        }
        const int localRow = globalRow - localBBlock.firstRow;
        const CsrOffset nonZeros = localMatrixB.rowPtr[localRow + 1] - localMatrixB.rowPtr[localRow];
        rowNonZeros[index] = nonZeros;
        totalNonZeros = checkedCsrAdd(totalNonZeros, nonZeros, "halo payload");
    }

    columns.resize(csrSize<int>(totalNonZeros));
    values.resize(csrSize<double>(totalNonZeros));
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
    for (int peer = 0; peer < checkedMatrixDimension(checkedCsrCount(blocks.size())); ++peer) {
        auto& rows = plan.rowsByPeer[peer];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        plan.peerOffsets[peer] = totalRemoteRows;
        totalRemoteRows += checkedMatrixDimension(checkedCsrCount(rows.size()));
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

TwoSidedPlan buildTwoSidedPlan(const CsrMatrix& localMatrixA, const CsrMatrix& localMatrixB,
                              const std::vector<RowBlock>& bBlocks, RowBlock localBBlock,
                              int rank, int ranks, MPI_Comm communicator) {
    RemoteSparseRows remoteRows =
        makeRemoteSparseRows(buildRemoteRowPlan(localMatrixA, bBlocks, rank, communicator));

    std::vector<int> incomingCounts(ranks, 0);
    std::vector<int> outgoingCounts(ranks, 0);
    for (int peer = 0; peer < ranks; ++peer) {
        outgoingCounts[peer] = checkedMatrixDimension(checkedCsrCount(remoteRows.plan.rowsByPeer[peer].size()));
    }

    std::vector<MPI_Request> requests;
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        csrIrecv(&incomingCounts[peer], 1, peer, kRequestCountTag, communicator, requests);
        csrIsend(&outgoingCounts[peer], 1, peer, kRequestCountTag, communicator, requests);
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

        csrIrecv(dataOrNull(incomingRequests[peer]), incomingCounts[peer], peer, kRequestRowsTag, communicator, requests);
        csrIsend(dataOrNull(remoteRows.plan.rowsByPeer[peer]), checkedMatrixDimension(checkedCsrCount(remoteRows.plan.rowsByPeer[peer].size())), peer, kRequestRowsTag, communicator, requests);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<CsrOffset>> incomingRowNonZeros(ranks);
    std::vector<std::vector<CsrOffset>> outgoingRowNonZeros(ranks);
    std::vector<std::vector<int>> outgoingColumns(ranks);
    std::vector<std::vector<double>> outgoingValues(ranks);
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        incomingRowNonZeros[peer].resize(outgoingCounts[peer]);
        prepareRequestedRows(localMatrixB, localBBlock, incomingRequests[peer],
                          outgoingRowNonZeros[peer], outgoingColumns[peer], outgoingValues[peer],
                          communicator);
    }

    requests.clear();
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        csrIrecv(dataOrNull(incomingRowNonZeros[peer]), outgoingCounts[peer], peer, kSparseRowNnzTag, communicator, requests);
        csrIsend(dataOrNull(outgoingRowNonZeros[peer]), checkedMatrixDimension(checkedCsrCount(outgoingRowNonZeros[peer].size())), peer, kSparseRowNnzTag, communicator, requests);
    }
    waitAll(requests, communicator);

    CsrOffset totalRemoteNonZeros = 0;
    for (int peer = 0; peer < ranks; ++peer) {
        const int peerOffset = remoteRows.plan.peerOffsets[peer];
        for (int index = 0; index < checkedMatrixDimension(checkedCsrCount(incomingRowNonZeros[peer].size())); ++index) {
            remoteRows.rowPtr[peerOffset + index] = totalRemoteNonZeros;
            totalRemoteNonZeros = checkedCsrAdd(totalRemoteNonZeros, incomingRowNonZeros[peer][index], "halo prefix");
            remoteRows.rowPtr[peerOffset + index + 1] = totalRemoteNonZeros;
        }
    }
    remoteRows.columnIndices.resize(csrSize<int>(totalRemoteNonZeros));
    remoteRows.values.resize(csrSize<double>(totalRemoteNonZeros));

    return {std::move(remoteRows), std::move(incomingRequests), std::move(outgoingColumns),
            std::move(outgoingValues)};
}

void exchangeRemoteSparseRowsTwoSided(const CsrMatrix& localMatrixB, RowBlock localBBlock,
                                     TwoSidedPlan& plan, int rank, MPI_Comm communicator) {
    auto& remoteRows = plan.remoteRows;
    auto& outgoingColumns = plan.outgoingColumns;
    auto& outgoingValues = plan.outgoingValues;
    const int ranks = checkedMatrixDimension(checkedCsrCount(plan.incomingRequests.size()));
    for (int peer = 0; peer < ranks; ++peer) {
        CsrOffset offset = 0;
        for (const int globalRow : plan.incomingRequests[peer]) {
            const int localRow = globalRow - localBBlock.firstRow;
            const CsrOffset first = localMatrixB.rowPtr[localRow];
            const CsrOffset count = localMatrixB.rowPtr[localRow + 1] - first;
            std::copy_n(localMatrixB.columnIndices.begin() + first, count,
                        outgoingColumns[peer].begin() + offset);
            std::copy_n(localMatrixB.values.begin() + first, count,
                        outgoingValues[peer].begin() + offset);
            offset = checkedCsrAdd(offset, count, "halo packing offset");
        }
    }

    std::vector<MPI_Request> requests;
    requests.reserve(4 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        const CsrOffset receiveOffset = remoteRows.rowPtr[remoteRows.plan.peerOffsets[peer]];
        const CsrOffset receiveCount =
            remoteRows.rowPtr[remoteRows.plan.peerOffsets[peer + 1]] - receiveOffset;
        const CsrOffset sendCount = checkedCsrCount(outgoingColumns[peer].size());

        csrIrecv(dataAtOrNull(remoteRows.columnIndices, receiveOffset, receiveCount), receiveCount, peer, kSparseColumnTag, communicator, requests);
        csrIrecv(dataAtOrNull(remoteRows.values, receiveOffset, receiveCount), receiveCount, peer, kSparseValueTag, communicator, requests);
        csrIsend(dataOrNull(outgoingColumns[peer]), sendCount, peer, kSparseColumnTag, communicator, requests);
        csrIsend(dataOrNull(outgoingValues[peer]), sendCount, peer, kSparseValueTag, communicator, requests);
    }
    waitAll(requests, communicator);

}

CsrMatrix spgemm(const CsrMatrix& localMatrixA, RowBlock localBBlock,
                 const CsrMatrix& localMatrixB, const RemoteSparseRows& remoteBRows,
                 MPI_Comm communicator) {
    CsrMatrix result;
    result.rows = localMatrixA.rows;
    result.cols = localMatrixB.cols;
    result.rowPtr.resize(static_cast<std::size_t>(localMatrixA.rows) + 1, 0);

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

            for (CsrOffset aEntry = localMatrixA.rowPtr[row]; aEntry < localMatrixA.rowPtr[row + 1];
                 ++aEntry) {
                const int bRow = localMatrixA.columnIndices[aEntry];
                const double aValue = localMatrixA.values[aEntry];
                const SparseRowView bValues =
                    bRowForColumn(localMatrixB, localBBlock, remoteBRows, bRow, communicator);
                for (CsrOffset bEntry = 0; bEntry < bValues.nonZeros; ++bEntry) {
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

    CsrOffset totalNonZeros = 0;
    for (int row = 0; row < result.rows; ++row) {
        totalNonZeros = checkedCsrAdd(totalNonZeros, checkedCsrCount(valuesByRow[row].size()), "local result prefix");
        result.rowPtr[row + 1] = totalNonZeros;
    }
    result.columnIndices.reserve(csrSize<int>(totalNonZeros));
    result.values.reserve(csrSize<double>(totalNonZeros));
    for (int row = 0; row < result.rows; ++row) {
        result.columnIndices.insert(result.columnIndices.end(), columnsByRow[row].begin(),
                                    columnsByRow[row].end());
        result.values.insert(result.values.end(), valuesByRow[row].begin(), valuesByRow[row].end());
    }
    return result;
}
