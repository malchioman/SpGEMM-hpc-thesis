#include "spmm_exchange.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace {

constexpr int kRequestCountTag = 200;
constexpr int kRequestRowsTag = 201;
constexpr int kDenseRowsTag = 202;

template <typename T>
T* dataOrNull(std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

template <typename T>
const T* dataOrNull(const std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

}  // namespace

RemoteRowPlan buildRemoteRowPlan(const CsrMatrix& localMatrix, const std::vector<RowBlock>& blocks,
                                 int rank, MPI_Comm communicator) {
    RemoteRowPlan plan;
    plan.rowsByPeer.resize(blocks.size());

    for (const int column : localMatrix.columnIndices) {
        const int owner = ownerOfRow(column, blocks);
        if (owner < 0) {
            fail("sparse matrix contains an invalid column index", communicator);
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

RemoteDenseRows makeRemoteDenseRows(RemoteRowPlan plan, int denseCols) {
    RemoteDenseRows remoteDenseRows;
    remoteDenseRows.plan = std::move(plan);
    remoteDenseRows.values.assign(
        static_cast<std::size_t>(remoteDenseRows.plan.flatRows.size()) * denseCols, 0.0);
    return remoteDenseRows;
}

RemoteDenseRows exchangeRemoteDenseRowsTwoSided(const CsrMatrix& localMatrix,
                                                const std::vector<RowBlock>& blocks,
                                                RowBlock localBlock,
                                                const std::vector<double>& ownedDenseRows,
                                                int denseCols, int rank, int ranks,
                                                MPI_Comm communicator) {
    RemoteDenseRows remoteDenseRows =
        makeRemoteDenseRows(buildRemoteRowPlan(localMatrix, blocks, rank, communicator), denseCols);

    std::vector<int> incomingCounts(ranks, 0);
    std::vector<int> outgoingCounts(ranks, 0);
    for (int peer = 0; peer < ranks; ++peer) {
        outgoingCounts[peer] = static_cast<int>(remoteDenseRows.plan.rowsByPeer[peer].size());
    }

    std::vector<MPI_Request> requests;
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }
        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(&incomingCounts[peer], 1, MPI_INT, peer, kRequestCountTag, communicator,
                           &request), "MPI_Irecv(request count)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(&outgoingCounts[peer], 1, MPI_INT, peer, kRequestCountTag, communicator,
                           &request), "MPI_Isend(request count)", communicator);
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
        checkMpi(MPI_Isend(dataOrNull(remoteDenseRows.plan.rowsByPeer[peer]),
                           static_cast<int>(remoteDenseRows.plan.rowsByPeer[peer].size()), MPI_INT,
                           peer, kRequestRowsTag, communicator, &request),
                 "MPI_Isend(row requests)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    std::vector<std::vector<double>> outgoingDenseRows(ranks);
    requests.clear();
    requests.reserve(2 * (ranks - 1));
    for (int peer = 0; peer < ranks; ++peer) {
        if (peer == rank) {
            continue;
        }

        const int receiveRows = static_cast<int>(remoteDenseRows.plan.rowsByPeer[peer].size());
        const int receiveCount = receiveRows * denseCols;
        double* receiveBuffer = nullptr;
        if (receiveCount > 0) {
            receiveBuffer = remoteDenseRows.values.data() +
                            static_cast<std::size_t>(remoteDenseRows.plan.peerOffsets[peer]) *
                                denseCols;
        }

        outgoingDenseRows[peer].resize(static_cast<std::size_t>(incomingRequests[peer].size()) *
                                       denseCols);
        for (int index = 0; index < static_cast<int>(incomingRequests[peer].size()); ++index) {
            const int globalRow = incomingRequests[peer][index];
            if (globalRow < localBlock.firstRow || globalRow >= localBlock.firstRow + localBlock.rows) {
                fail("received a request for a non-local dense row", communicator);
            }
            const int localRow = globalRow - localBlock.firstRow;
            std::copy_n(ownedDenseRows.data() + static_cast<std::size_t>(localRow) * denseCols,
                        denseCols,
                        outgoingDenseRows[peer].data() + static_cast<std::size_t>(index) * denseCols);
        }

        MPI_Request request = MPI_REQUEST_NULL;
        checkMpi(MPI_Irecv(receiveBuffer, receiveCount, MPI_DOUBLE, peer, kDenseRowsTag, communicator,
                           &request), "MPI_Irecv(dense rows)", communicator);
        requests.push_back(request);
        checkMpi(MPI_Isend(dataOrNull(outgoingDenseRows[peer]),
                           static_cast<int>(outgoingDenseRows[peer].size()), MPI_DOUBLE, peer,
                           kDenseRowsTag, communicator, &request),
                 "MPI_Isend(dense rows)", communicator);
        requests.push_back(request);
    }
    waitAll(requests, communicator);

    return remoteDenseRows;
}

void spmm(const CsrMatrix& matrix, RowBlock localBlock, const std::vector<double>& ownedDenseRows,
          const RemoteDenseRows& remoteDenseRows, int denseCols, std::vector<double>& result) {
    std::fill(result.begin(), result.end(), 0.0);
#pragma omp parallel for schedule(runtime)
    for (int row = 0; row < matrix.rows; ++row) {
        double* output = result.data() + static_cast<std::size_t>(row) * denseCols;
        for (int entry = matrix.rowPtr[row]; entry < matrix.rowPtr[row + 1]; ++entry) {
            const int denseRow = matrix.columnIndices[entry];
            const double* input = nullptr;
            if (denseRow >= localBlock.firstRow && denseRow < localBlock.firstRow + localBlock.rows) {
                input = ownedDenseRows.data() +
                        static_cast<std::size_t>(denseRow - localBlock.firstRow) * denseCols;
            } else {
                const int slot = remoteDenseRows.plan.rowToSlot.at(denseRow);
                input = remoteDenseRows.values.data() + static_cast<std::size_t>(slot) * denseCols;
            }
            const double value = matrix.values[entry];
            for (int column = 0; column < denseCols; ++column) {
                output[column] += value * input[column];
            }
        }
    }
}
