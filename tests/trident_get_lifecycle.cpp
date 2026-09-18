#include "inter_node_exchange.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

template <typename F>
void expectRejected(F operation, const char* description) {
    try {
        operation();
    } catch (const std::logic_error&) {
        return;
    }
    throw std::runtime_error(description);
}

void update(CsrMatrix& matrix, int owner, int generation, int salt) {
    matrix.rowPtr[1] = (generation + salt) % 3;
    matrix.columnIndices[0] = (generation + salt) % 3;
    matrix.columnIndices[1] = (generation + salt + 1) % 3;
    matrix.values[0] = 1000 * generation + 10 * owner + salt;
    matrix.values[1] = -matrix.values[0] - 1;
}

void checkTile(const CsrMatrix& actual, int owner, int generation, int salt) {
    CsrMatrix expected{2, 3, {0, 1, 2}, {0, 1}, {0, 0}};
    update(expected, owner, generation, salt);
    if (actual.rows != expected.rows || actual.cols != expected.cols ||
        actual.rowPtr != expected.rowPtr || actual.columnIndices != expected.columnIndices ||
        actual.values != expected.values) {
        throw std::runtime_error("GET did not fetch the current complete CSR payload");
    }
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return EXIT_FAILURE;
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required", MPI_COMM_WORLD);
    try {
        trident::ProcessGrid grid(MPI_COMM_WORLD, argc > 1 ? std::stoi(argv[1]) : 1);
        CsrMatrix a{2, 3, {0, 1, 2}, {0, 1}, {0, 0}};
        CsrMatrix b = a;
        trident::Stage stage;
        stage.a = {2, 3, 2};
        stage.b = {2, 3, 2};
        stage.aSource = grid.rankAt(grid.row, (grid.col + 1) % grid.side, grid.nodeRank);
        stage.bSource = grid.rankAt((grid.row + 1) % grid.side, grid.col, grid.nodeRank);
        trident::ExecutionPlan plan;
        plan.stages.push_back(stage);
        trident::ProductWorkspace workspace(plan);
        trident::GetInterNodeExchange exchange(grid, a, b);
        expectRejected([&] { exchange.fetch(a, b, stage, workspace); }, "fetch before begin accepted");
        expectRejected([&] { exchange.finish(); }, "finish before begin accepted");
        const auto otherA = a;
        expectRejected([&] { exchange.begin(otherA, b); }, "unbound input buffers accepted");
        ++a.cols;
        expectRejected([&] { exchange.begin(a, b); }, "changed input shape accepted");
        --a.cols;

        for (int generation = 0; generation < 24; ++generation) {
            // Every CSR array changes in place; no external barriers protect successive products.
            std::this_thread::sleep_for(std::chrono::milliseconds((grid.rank + generation) % 4));
            update(a, grid.rank, generation, 1);
            update(b, grid.rank, generation, 2);
            exchange.begin(a, b);
            expectRejected([&] { exchange.begin(a, b); }, "nested product accepted");
            expectRejected([&] { exchange.close(); }, "close during product accepted");
            exchange.fetch(a, b, stage, workspace);
            checkTile(workspace.a, stage.aSource, generation, 1);
            checkTile(workspace.b, stage.bSource, generation, 2);
            exchange.finish();
        }
        expectRejected([&] { exchange.finish(); }, "duplicate finish accepted");
        exchange.close();
        expectRejected([&] { exchange.begin(a, b); }, "begin after close accepted");
        expectRejected([&] { exchange.close(); }, "duplicate close accepted");
        trident::GetTileTransport transfer(grid, a, b);
        expectRejected([&] { transfer.completeFetch(); }, "completion without a product accepted");
        transfer.begin(a, b);
        expectRejected([&] { transfer.completeFetch(); }, "completion without a fetch accepted");
        transfer.startFetch(a, b, stage, workspace.a, workspace.b);
        expectRejected([&] { transfer.startFetch(a, b, stage, workspace.a, workspace.b); },
                       "overlapping use of GET destination buffers accepted");
        expectRejected([&] { transfer.finish(); }, "finish with outstanding GET accepted");
        transfer.completeFetch();
        checkTile(workspace.a, stage.aSource, 23, 1);
        checkTile(workspace.b, stage.bSource, 23, 2);
        expectRejected([&] { transfer.completeFetch(); }, "duplicate completion accepted");
        transfer.finish();
        transfer.close();
        if (grid.rank == 0) std::cout << "GET full-CSR freshness, input binding and lifecycle: PASS\n";
        grid.close();
    } catch (const std::exception& error) {
        fail(error.what(), MPI_COMM_WORLD);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}
