#include "execution.hpp"

#include <algorithm>
#include <stdexcept>

namespace trident {
namespace {

int wrap(int value, int size) { return (value % size + size) % size; }

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

void exchangeInterNode(const CsrMatrix& a, const CsrMatrix& b, const ProcessGrid& grid,
                       const Stage& stage, ProductWorkspace& workspace) {
    resizeTile(workspace.a, stage.a);
    resizeTile(workspace.b, stage.b);
    std::vector<MPI_Request> requests;
    requests.reserve(12);
    if (stage.aSource == grid.rank) {
        workspace.a = a;
    } else {
        receiveTile(workspace.a, stage.aSource, 20, grid.world, requests);
    }
    if (stage.bSource == grid.rank) {
        workspace.b = b;
    } else {
        receiveTile(workspace.b, stage.bSource, 30, grid.world, requests);
    }
    // Invert h=(i+j+round)%q: each static owner has exactly one consumer per round.
    if (stage.aTarget != grid.rank) {
        sendTile(a, stage.aTarget, 20, grid.world, requests);
    }
    if (stage.bTarget != grid.rank) {
        sendTile(b, stage.bTarget, 30, grid.world, requests);
    }
    waitAll(requests, grid.world);
}

}  // namespace

ExecutionPlan preparePlan(const CsrMatrix& a, const CsrMatrix& b, const ProcessGrid& grid) {
    const std::array<int, 6> local{a.rows, a.cols, mpiCount(a.values.size()),
                                  b.rows, b.cols, mpiCount(b.values.size())};
    std::vector<std::array<int, 6>> shapes(grid.size);
    checkMpi(MPI_Allgather(local.data(), 6, MPI_INT, shapes.data(), 6, MPI_INT, grid.world),
             "MPI_Allgather(tile shapes)", grid.world);
    ExecutionPlan plan;
    plan.resultRows = a.rows;
    plan.resultCols = b.cols;
    for (int round = 0; round < grid.side; ++round) {
        Stage stage;
        stage.inner = (grid.row + grid.col + round) % grid.side;
        stage.aSource = grid.rankAt(grid.row, stage.inner, grid.nodeRank);
        stage.bSource = grid.rankAt(stage.inner, grid.col, grid.nodeRank);
        stage.aTarget = grid.rankAt(grid.row, wrap(grid.col - grid.row - round, grid.side), grid.nodeRank);
        stage.bTarget = grid.rankAt(wrap(grid.row - grid.col - round, grid.side), grid.col, grid.nodeRank);
        const auto& sa = shapes[stage.aSource];
        const auto& sb = shapes[stage.bSource];
        stage.a = {sa[0], sa[1], sa[2]};
        stage.b = {sb[3], sb[4], sb[5]};
        stage.panel.cols = b.cols;
        for (int k = 0; k < grid.nodeSize; ++k) {
            const auto& shape = shapes[grid.rankAt(stage.inner, grid.col, k)];
            stage.rowCounts.push_back(shape[3]);
            stage.rowOffsets.push_back(stage.panel.rows);
            stage.nnzCounts.push_back(shape[5]);
            stage.nnzOffsets.push_back(stage.panel.nnz);
            stage.panel.rows = mpiCount(static_cast<std::size_t>(stage.panel.rows) + shape[3]);
            stage.panel.nnz = mpiCount(static_cast<std::size_t>(stage.panel.nnz) + shape[5]);
            if (shape[4] != stage.panel.cols) {
                throw std::runtime_error("inconsistent B column slices within a Trident node");
            }
        }
        if (stage.a.rows != plan.resultRows || stage.a.cols != stage.panel.rows) {
            throw std::runtime_error("incompatible Trident stage shapes");
        }
        plan.stages.push_back(std::move(stage));
    }
    return plan;
}

ProductWorkspace::ProductWorkspace(const ExecutionPlan& plan) {
    auto reserve = [](CsrMatrix& matrix, TileShape shape) {
        matrix.rowPtr.reserve(static_cast<std::size_t>(shape.rows) + 1);
        matrix.columnIndices.reserve(shape.nnz);
        matrix.values.reserve(shape.nnz);
    };
    for (const auto& stage : plan.stages) {
        reserve(a, stage.a);
        reserve(b, stage.b);
        reserve(panel, stage.panel);
    }
}

ProductResult multiply(const CsrMatrix& a, const CsrMatrix& b, const ProcessGrid& grid,
                       const ExecutionPlan& plan, IntraNodeExchange& exchange,
                       ProductWorkspace& workspace) {
    ProductResult result;
    const double start = MPI_Wtime();
    double finishStart = 0.0;
    {
        LocalAccumulator accumulator(plan.resultRows, plan.resultCols);
        result.computeSeconds = MPI_Wtime() - start;
        for (const auto& stage : plan.stages) {
            const double interStart = MPI_Wtime();
            exchangeInterNode(a, b, grid, stage, workspace);
            const double intraStart = MPI_Wtime();
            exchange.assemble(workspace.b, stage, workspace.panel);
            const double computeStart = MPI_Wtime();
            accumulator.addProduct(workspace.a, workspace.panel);
            const double end = MPI_Wtime();
            result.interNodeSeconds += intraStart - interStart;
            result.intraNodeSeconds += computeStart - intraStart;
            result.computeSeconds += end - computeStart;
        }
        finishStart = MPI_Wtime();
        result.matrix = accumulator.finish();
    }
    // Include destruction of the per-row hash tables in the compute measurement.
    result.computeSeconds += MPI_Wtime() - finishStart;
    result.totalSeconds = MPI_Wtime() - start;
    return result;
}

}  // namespace trident
