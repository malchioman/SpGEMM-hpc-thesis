#include "inter_node_exchange.hpp"

#include "csr_transfer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace trident {
namespace {

template <typename T>
MPI_Win exposeBuffer(std::vector<T>& buffer, MPI_Comm comm) {
    mpiCount(buffer.size());
    if (buffer.size() > static_cast<std::size_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(T)) {
        throw std::overflow_error("PUT window size exceeds MPI_Aint");
    }
    MPI_Win window = MPI_WIN_NULL;
    checkMpi(MPI_Win_create(buffer.empty() ? nullptr : buffer.data(),
                            static_cast<MPI_Aint>(buffer.size() * sizeof(T)), sizeof(T),
                            MPI_INFO_NULL, comm, &window), "MPI_Win_create(PUT receiver)", comm);
    checkMpi(MPI_Win_lock_all(0, window), "MPI_Win_lock_all(PUT receiver)", comm);
    return window;
}

void checkShape(const CsrMatrix& input, TileShape shape) {
    if (input.rows < 0 || input.cols < 0 || input.rows != shape.rows || input.cols != shape.cols ||
        input.rowPtr.size() != static_cast<std::size_t>(input.rows) + 1 ||
        input.columnIndices.size() != input.values.size() || mpiCount(input.values.size()) != shape.nnz) {
        throw std::invalid_argument("PUT inputs must retain their planned shapes and CSR array sizes");
    }
}

}  // namespace

PutInterNodeExchange::PutInterNodeExchange(const ProcessGrid& grid, const ExecutionPlan& plan,
                                         const CsrMatrix& a, const CsrMatrix& b, ProductWorkspace& workspace)
    : grid_(grid), plan_(plan), workspace_(workspace),
      aShape_{a.rows, a.cols, mpiCount(a.values.size())},
      bShape_{b.rows, b.cols, mpiCount(b.values.size())} {
    checkInputs(a, b);
    aWindow_ = expose(workspace_.a, true);
    bWindow_ = expose(workspace_.b, false);
}

PutInterNodeExchange::~PutInterNodeExchange() {
    if (!closed_) fail("PUT backend destroyed before collective close", grid_.world);
}

PutInterNodeExchange::ReceiveWindow PutInterNodeExchange::expose(CsrMatrix& tile, bool operandA) {
    ReceiveWindow result{};
    for (const auto& stage : plan_.stages) {
        const auto shape = operandA ? stage.a : stage.b;
        result.maximum.rows = std::max(result.maximum.rows, shape.rows);
        result.maximum.cols = std::max(result.maximum.cols, shape.cols);
        result.maximum.nnz = std::max(result.maximum.nnz, shape.nnz);
    }
    // Expose the largest stage once. Later resizes stay within this allocation.
    resizeTile(tile, result.maximum);
    result.bases = {tile.rowPtr.data(), tile.columnIndices.data(), tile.values.data()};
    if (grid_.side > 1) {
        result.windows[0] = exposeBuffer(tile.rowPtr, grid_.world);
        result.windows[1] = exposeBuffer(tile.columnIndices, grid_.world);
        result.windows[2] = exposeBuffer(tile.values, grid_.world);
    }
    return result;
}

void PutInterNodeExchange::checkWorkspace() const {
    const auto check = [](const CsrMatrix& tile, const ReceiveWindow& receiver) {
        if (tile.rowPtr.data() != receiver.bases[0] || tile.columnIndices.data() != receiver.bases[1] ||
            tile.values.data() != receiver.bases[2] ||
            tile.rowPtr.capacity() < static_cast<std::size_t>(receiver.maximum.rows) + 1 ||
            tile.columnIndices.capacity() < static_cast<std::size_t>(receiver.maximum.nnz) ||
            tile.values.capacity() < static_cast<std::size_t>(receiver.maximum.nnz)) {
            throw std::invalid_argument("PUT workspace must retain its exposed buffer addresses and capacity");
        }
    };
    check(workspace_.a, aWindow_);
    check(workspace_.b, bWindow_);
}

void PutInterNodeExchange::checkInputs(const CsrMatrix& a, const CsrMatrix& b) const {
    checkShape(a, aShape_);
    checkShape(b, bShape_);
}

void PutInterNodeExchange::syncReceivers() {
    for (const auto* receiver : {&aWindow_, &bWindow_}) {
        for (const auto window : receiver->windows) {
            checkMpi(MPI_Win_sync(window), "MPI_Win_sync(PUT receiver)", grid_.world);
        }
    }
}

void PutInterNodeExchange::begin(const CsrMatrix& a, const CsrMatrix& b) {
    if (active_ || closed_) throw std::logic_error("invalid PUT product lifecycle");
    checkInputs(a, b);
    checkWorkspace();
    nextStage_ = 0;
    active_ = true;
}

void PutInterNodeExchange::putTile(const CsrMatrix& tile, int target, const ReceiveWindow& receiver) {
    checkMpi(MPI_Put(tile.rowPtr.data(), mpiCount(tile.rowPtr.size()), MPI_INT, target, 0,
                     mpiCount(tile.rowPtr.size()), MPI_INT, receiver.windows[0]),
             "MPI_Put(tile pointers)", grid_.world);
    const int nnz = mpiCount(tile.values.size());
    if (nnz > 0) {
        checkMpi(MPI_Put(tile.columnIndices.data(), nnz, MPI_INT, target, 0, nnz, MPI_INT,
                         receiver.windows[1]), "MPI_Put(tile indices)", grid_.world);
        checkMpi(MPI_Put(tile.values.data(), nnz, MPI_DOUBLE, target, 0, nnz, MPI_DOUBLE,
                         receiver.windows[2]), "MPI_Put(tile values)", grid_.world);
    }
}

void PutInterNodeExchange::complete(int target, const ReceiveWindow& receiver) {
    for (const auto window : receiver.windows) {
        checkMpi(MPI_Win_flush(target, window), "MPI_Win_flush(PUT tile)", grid_.world);
    }
}

void PutInterNodeExchange::fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                                ProductWorkspace& workspace) {
    if (!active_ || nextStage_ >= plan_.stages.size() || &stage != &plan_.stages[nextStage_]) {
        throw std::logic_error("PUT requires the next stage of its active bound plan");
    }
    if (&workspace != &workspace_) throw std::invalid_argument("PUT requires its bound workspace");
    checkInputs(a, b);
    checkWorkspace();
    resizeTile(workspace.a, stage.a);
    resizeTile(workspace.b, stage.b);
    // Copy into existing storage; assignment must not replace an exposed allocation.
    const auto copyLocal = [](const CsrMatrix& input, CsrMatrix& output) {
        std::copy(input.rowPtr.begin(), input.rowPtr.end(), output.rowPtr.begin());
        std::copy(input.columnIndices.begin(), input.columnIndices.end(), output.columnIndices.begin());
        std::copy(input.values.begin(), input.values.end(), output.values.begin());
    };
    if (stage.aSource == grid_.rank) copyLocal(a, workspace.a);
    if (stage.bSource == grid_.rank) copyLocal(b, workspace.b);
    if (grid_.side > 1) {
        // Readers have finished and any resize/local stores precede new remote writes.
        syncReceivers();
        checkMpi(MPI_Barrier(grid_.world), "MPI_Barrier(PUT receivers ready)", grid_.world);
        if (stage.aTarget != grid_.rank) putTile(a, stage.aTarget, aWindow_);
        if (stage.bTarget != grid_.rank) putTile(b, stage.bTarget, bWindow_);
        if (stage.aTarget != grid_.rank) complete(stage.aTarget, aWindow_);
        if (stage.bTarget != grid_.rank) complete(stage.bTarget, bWindow_);
        // A barrier alone does not complete RMA or synchronize private/public copies.
        checkMpi(MPI_Barrier(grid_.world), "MPI_Barrier(PUT writers done)", grid_.world);
        syncReceivers();
    }
    ++nextStage_;
}

void PutInterNodeExchange::finish() {
    if (!active_ || nextStage_ != plan_.stages.size()) {
        throw std::logic_error("PUT finish before consuming all stages of an active product");
    }
    active_ = false;
}

void PutInterNodeExchange::close() {
    if (active_ || closed_) throw std::logic_error("invalid PUT close");
    checkWorkspace();
    for (auto* receiver : {&aWindow_, &bWindow_}) {
        for (auto& window : receiver->windows) {
            if (window == MPI_WIN_NULL) continue;
            checkMpi(MPI_Win_unlock_all(window), "MPI_Win_unlock_all(PUT receiver)", grid_.world);
            checkMpi(MPI_Win_free(&window), "MPI_Win_free(PUT receiver)", grid_.world);
        }
    }
    closed_ = true;
}

}  // namespace trident
