#include "inter_node_exchange.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace audit {

struct Read {
    void* destination;
    int source;
    MPI_Win window;
};

bool enabled = false;
int issued = 0;
int flushed = 0;
std::vector<Read> pending;

}  // namespace audit

// PMPI observes actual issue/completion order without relying on network timing.
extern "C" int MPI_Get(void* destination, int count, MPI_Datatype datatype, int source,
                       MPI_Aint displacement, int remoteCount, MPI_Datatype remoteType, MPI_Win window) {
    const int result = PMPI_Get(destination, count, datatype, source, displacement,
                               remoteCount, remoteType, window);
    if (result == MPI_SUCCESS && audit::enabled) {
        ++audit::issued;
        audit::pending.push_back({destination, source, window});
    }
    return result;
}

extern "C" int MPI_Win_flush(int source, MPI_Win window) {
    const int result = PMPI_Win_flush(source, window);
    if (result == MPI_SUCCESS && audit::enabled) {
        ++audit::flushed;
        auto& pending = audit::pending;
        pending.erase(std::remove_if(pending.begin(), pending.end(), [=](const audit::Read& read) {
            return read.source == source && read.window == window;
        }), pending.end());
    }
    return result;
}

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
        throw std::runtime_error("pipeline returned a stale or incorrect CSR payload");
    }
}

int remoteCalls(const trident::Stage& stage, int rank) {
    return 3 * ((stage.aSource != rank) + (stage.bSource != rank));
}

void checkTrace(int issued, int flushed, int pending, const trident::ProductWorkspace& workspace) {
    if (audit::issued != issued || audit::flushed != flushed ||
        audit::pending.size() != static_cast<std::size_t>(pending)) {
        throw std::runtime_error("pipeline did not defer completion or prefetch exactly one stage");
    }
    const std::array<const void*, 6> current{workspace.a.rowPtr.data(), workspace.a.columnIndices.data(),
        workspace.a.values.data(), workspace.b.rowPtr.data(), workspace.b.columnIndices.data(),
        workspace.b.values.data()};
    for (const auto& read : audit::pending) {
        if (std::find(current.begin(), current.end(), read.destination) != current.end()) {
            throw std::runtime_error("pending GET overwrites a buffer used by the current stage");
        }
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
        trident::ExecutionPlan plan;
        for (int round = 0; round < grid.side; ++round) {
            trident::Stage stage;
            stage.inner = (grid.row + grid.col + round) % grid.side;
            stage.a = stage.b = {2, 3, 2};
            stage.aSource = grid.rankAt(grid.row, stage.inner, grid.nodeRank);
            stage.bSource = grid.rankAt(stage.inner, grid.col, grid.nodeRank);
            plan.stages.push_back(stage);
        }
        trident::ProductWorkspace workspace(plan);
        trident::PipelinedGetInterNodeExchange exchange(grid, plan, a, b);
        expectRejected([&] { exchange.fetch(a, b, plan.stages.front(), workspace); },
                       "fetch before begin accepted");
        expectRejected([&] { exchange.finish(); }, "finish before begin accepted");
        const auto otherA = a;
        expectRejected([&] { exchange.begin(otherA, b); }, "unbound input buffers accepted");
        ++a.cols;
        expectRejected([&] { exchange.begin(a, b); }, "changed input shape accepted");
        --a.cols;

        for (int generation = 0; generation < 24; ++generation) {
            std::this_thread::sleep_for(std::chrono::milliseconds((grid.rank + generation) % 4));
            update(a, grid.rank, generation, 1);
            update(b, grid.rank, generation, 2);
            audit::issued = audit::flushed = 0;
            audit::pending.clear();
            audit::enabled = true;
            exchange.begin(a, b);
            expectRejected([&] { exchange.begin(a, b); }, "nested product accepted");
            expectRejected([&] { exchange.close(); }, "close with a prefetched stage accepted");
            expectRejected([&] { exchange.finish(); }, "finish before consuming stages accepted");
            if (plan.stages.size() > 1) {
                expectRejected([&] { exchange.fetch(a, b, plan.stages[1], workspace); },
                               "out-of-order stage accepted");
            }
            int issued = remoteCalls(plan.stages.front(), grid.rank);
            int flushed = 0;
            checkTrace(issued, flushed, issued, workspace);
            std::vector<const double*> buffersA;
            std::vector<const double*> buffersB;
            for (std::size_t round = 0; round < plan.stages.size(); ++round) {
                const auto& stage = plan.stages[round];
                exchange.fetch(a, b, stage, workspace);
                flushed += remoteCalls(stage, grid.rank);
                const int pending = round + 1 < plan.stages.size()
                                        ? remoteCalls(plan.stages[round + 1], grid.rank) : 0;
                issued += pending;
                checkTrace(issued, flushed, pending, workspace);
                checkTile(workspace.a, stage.aSource, generation, 1);
                checkTile(workspace.b, stage.bSource, generation, 2);
                buffersA.push_back(workspace.a.values.data());
                buffersB.push_back(workspace.b.values.data());
                if (round > 0 && (buffersA[round] == buffersA[round - 1] ||
                                  buffersB[round] == buffersB[round - 1])) {
                    throw std::runtime_error("pipeline did not alternate input buffers");
                }
                if (round > 1 && (buffersA[round] != buffersA[round - 2] ||
                                  buffersB[round] != buffersB[round - 2])) {
                    throw std::runtime_error("pipeline allocated more than two input buffer pairs");
                }
                expectRejected([&] { exchange.fetch(a, b, stage, workspace); }, "duplicate stage accepted");
                std::this_thread::sleep_for(std::chrono::milliseconds((grid.rank + stage.inner) % 3));
                checkTile(workspace.a, stage.aSource, generation, 1);
                checkTile(workspace.b, stage.bSource, generation, 2);
            }
            exchange.finish();
            checkTrace(issued, flushed, 0, workspace);
            audit::enabled = false;
        }
        expectRejected([&] { exchange.finish(); }, "duplicate finish accepted");
        exchange.close();
        expectRejected([&] { exchange.begin(a, b); }, "begin after close accepted");
        expectRejected([&] { exchange.close(); }, "duplicate close accepted");
        if (grid.rank == 0) std::cout << "GET pipeline prefetch order, double buffering and freshness: PASS\n";
        grid.close();
    } catch (const std::exception& error) {
        fail(error.what(), MPI_COMM_WORLD);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}
