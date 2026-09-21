#include "inter_node_exchange.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace audit {

enum class Event { sync, barrier, put, flush };
struct Write {
    int target;
    MPI_Win window;
};
bool enabled = false;
bool barrierWithPending = false;
std::vector<Event> events;
std::vector<int> targets;
std::vector<Write> pending;

}  // namespace audit

// Observe actual calls, including the two synchronization gates around each stage.
extern "C" int MPI_Put(const void* source, int count, MPI_Datatype datatype, int target,
                       MPI_Aint displacement, int remoteCount, MPI_Datatype remoteType, MPI_Win window) {
    const int result = PMPI_Put(source, count, datatype, target, displacement, remoteCount, remoteType, window);
    if (result == MPI_SUCCESS && audit::enabled) {
        audit::events.push_back(audit::Event::put);
        audit::targets.push_back(target);
        audit::pending.push_back({target, window});
    }
    return result;
}

extern "C" int MPI_Win_flush(int target, MPI_Win window) {
    const int result = PMPI_Win_flush(target, window);
    if (result == MPI_SUCCESS && audit::enabled) {
        audit::events.push_back(audit::Event::flush);
        auto& pending = audit::pending;
        pending.erase(std::remove_if(pending.begin(), pending.end(), [=](const audit::Write& write) {
            return write.target == target && write.window == window;
        }), pending.end());
    }
    return result;
}

extern "C" int MPI_Win_sync(MPI_Win window) {
    const int result = PMPI_Win_sync(window);
    if (result == MPI_SUCCESS && audit::enabled) audit::events.push_back(audit::Event::sync);
    return result;
}

extern "C" int MPI_Barrier(MPI_Comm comm) {
    if (audit::enabled && !audit::pending.empty()) audit::barrierWithPending = true;
    const int result = PMPI_Barrier(comm);
    if (result == MPI_SUCCESS && audit::enabled) audit::events.push_back(audit::Event::barrier);
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
        throw std::runtime_error("PUT returned stale, overwritten or incorrect CSR payload");
    }
}

void checkTrace(const trident::Stage& stage, const trident::ProcessGrid& grid) {
    std::vector<audit::Event> expected;
    std::vector<int> targets;
    if (grid.side > 1) {
        if (stage.aTarget != grid.rank) targets.insert(targets.end(), 3, stage.aTarget);
        if (stage.bTarget != grid.rank) targets.insert(targets.end(), 3, stage.bTarget);
        expected.insert(expected.end(), 6, audit::Event::sync);
        expected.push_back(audit::Event::barrier);
        expected.insert(expected.end(), targets.size(), audit::Event::put);
        expected.insert(expected.end(), targets.size(), audit::Event::flush);
        expected.push_back(audit::Event::barrier);
        expected.insert(expected.end(), 6, audit::Event::sync);
    }
    if (audit::events != expected || audit::targets != targets ||
        !audit::pending.empty() || audit::barrierWithPending) {
        throw std::runtime_error("PUT stage readiness, destinations or completion order is incorrect");
    }
}

std::array<const void*, 6> bases(const trident::ProductWorkspace& workspace) {
    return {workspace.a.rowPtr.data(), workspace.a.columnIndices.data(), workspace.a.values.data(),
            workspace.b.rowPtr.data(), workspace.b.columnIndices.data(), workspace.b.values.data()};
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
        const auto wrap = [&](int x) { return (x % grid.side + grid.side) % grid.side; };
        for (int round = 0; round < grid.side; ++round) {
            trident::Stage stage;
            stage.inner = (grid.row + grid.col + round) % grid.side;
            stage.a = stage.b = {2, 3, 2};
            stage.aSource = grid.rankAt(grid.row, stage.inner, grid.nodeRank);
            stage.bSource = grid.rankAt(stage.inner, grid.col, grid.nodeRank);
            stage.aTarget = grid.rankAt(grid.row, wrap(grid.col - grid.row - round), grid.nodeRank);
            stage.bTarget = grid.rankAt(wrap(grid.row - grid.col - round), grid.col, grid.nodeRank);
            plan.stages.push_back(stage);
        }
        trident::ProductWorkspace workspace(plan);
        trident::ProductWorkspace otherWorkspace(plan);
        trident::PutInterNodeExchange exchange(grid, plan, a, b, workspace);
        const auto originalBases = bases(workspace);
        expectRejected([&] { exchange.fetch(a, b, plan.stages.front(), workspace); },
                       "fetch before begin accepted");
        expectRejected([&] { exchange.finish(); }, "finish before begin accepted");
        ++a.cols;
        expectRejected([&] { exchange.begin(a, b); }, "changed input shape accepted");
        --a.cols;
        auto badB = b;
        badB.values.push_back(3.0);
        expectRejected([&] { exchange.begin(a, badB); }, "changed CSR sizes accepted");

        for (int generation = 0; generation < 24; ++generation) {
            std::this_thread::sleep_for(std::chrono::milliseconds((grid.rank + generation) % 4));
            update(a, grid.rank, generation, 1);
            update(b, grid.rank, generation, 2);
            // PUT exposes receivers, not inputs: same-shape replacement input storage is valid.
            const auto inputA = a;
            const auto inputB = b;
            audit::events.clear();
            audit::targets.clear();
            audit::enabled = true;
            exchange.begin(inputA, inputB);
            if (!audit::events.empty()) throw std::runtime_error("PUT begin unexpectedly prefetched a stage");
            expectRejected([&] { exchange.begin(inputA, inputB); }, "nested product accepted");
            expectRejected([&] { exchange.close(); }, "close during a product accepted");
            expectRejected([&] { exchange.finish(); }, "finish before consuming stages accepted");
            expectRejected([&] { exchange.fetch(inputA, inputB, plan.stages.front(), otherWorkspace); },
                           "unbound workspace accepted");
            expectRejected([&] { exchange.fetch(inputA, badB, plan.stages.front(), workspace); },
                           "fetch with changed CSR sizes accepted");
            if (plan.stages.size() > 1) {
                expectRejected([&] { exchange.fetch(inputA, inputB, plan.stages[1], workspace); },
                               "out-of-order stage accepted");
            }
            for (const auto& stage : plan.stages) {
                audit::events.clear();
                audit::targets.clear();
                exchange.fetch(inputA, inputB, stage, workspace);
                checkTrace(stage, grid);
                if (bases(workspace) != originalBases) throw std::runtime_error("PUT receiver storage changed");
                checkTile(workspace.a, stage.aSource, generation, 1);
                checkTile(workspace.b, stage.bSource, generation, 2);
                expectRejected([&] { exchange.fetch(inputA, inputB, stage, workspace); },
                               "duplicate stage accepted");
                // Fast peers may enter the next stage, but cannot overwrite these arrays yet.
                std::this_thread::sleep_for(std::chrono::milliseconds((grid.rank + stage.inner) % 3));
                checkTile(workspace.a, stage.aSource, generation, 1);
                checkTile(workspace.b, stage.bSource, generation, 2);
            }
            audit::events.clear();
            exchange.finish();
            if (!audit::events.empty()) throw std::runtime_error("PUT left completion work after the final stage");
            audit::enabled = false;
        }
        expectRejected([&] { exchange.finish(); }, "duplicate finish accepted");
        exchange.close();
        expectRejected([&] { exchange.begin(a, b); }, "begin after close accepted");
        expectRejected([&] { exchange.close(); }, "duplicate close accepted");
        if (grid.rank == 0) std::cout << "PUT staged completion, receiver reuse and full-CSR freshness: PASS\n";
        grid.close();
    } catch (const std::exception& error) {
        fail(error.what(), MPI_COMM_WORLD);
    }
    checkMpi(MPI_Finalize(), "MPI_Finalize", MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}
