#include "execution.hpp"

#include "csr_transfer.hpp"

#include <algorithm>
#include <stdexcept>

namespace trident {
namespace {

int wrap(int value, int size) { return (value % size + size) % size; }

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
    for (const auto& stage : plan.stages) {
        reserveTile(a, stage.a);
        reserveTile(b, stage.b);
        reserveTile(panel, stage.panel);
    }
}

ProductResult multiply(const CsrMatrix& a, const CsrMatrix& b, const ExecutionPlan& plan,
                       IntraNodeExchange& exchange, ProductWorkspace& workspace,
                       InterNodeExchange& inter) {
    ProductResult result;
    const double start = MPI_Wtime();
    inter.begin(a, b);
    result.interNodeSeconds += MPI_Wtime() - start;
    const double computeInit = MPI_Wtime();
    double finishStart = 0.0;
    {
        LocalAccumulator accumulator(plan.resultRows, plan.resultCols);
        result.computeSeconds = MPI_Wtime() - computeInit;
        for (const auto& stage : plan.stages) {
            const double interStart = MPI_Wtime();
            inter.fetch(a, b, stage, workspace);
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
    const double finish = MPI_Wtime();
    inter.finish();
    result.interNodeSeconds += MPI_Wtime() - finish;
    result.totalSeconds = MPI_Wtime() - start;
    return result;
}

}  // namespace trident
