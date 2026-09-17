#include "inter_node_exchange.hpp"

#include "csr_transfer.hpp"

#include <limits>
#include <stdexcept>

namespace trident {
namespace {

template <typename T>
MPI_Win exposeBuffer(const std::vector<T>& buffer, MPI_Comm comm) {
    mpiCount(buffer.size());
    if (buffer.size() > static_cast<std::size_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(T)) {
        throw std::overflow_error("GET window size exceeds MPI_Aint");
    }
    MPI_Win window = MPI_WIN_NULL;
    // MPI_Win_create takes a mutable pointer, but this backend only exposes remote reads.
    void* base = buffer.empty() ? nullptr : const_cast<T*>(buffer.data());
    checkMpi(MPI_Win_create(base, static_cast<MPI_Aint>(buffer.size() * sizeof(T)), sizeof(T),
                            MPI_INFO_NULL, comm, &window), "MPI_Win_create(GET input)", comm);
    checkMpi(MPI_Win_lock_all(0, window), "MPI_Win_lock_all(GET input)", comm);
    return window;
}

}  // namespace

GetInterNodeExchange::GetInterNodeExchange(const ProcessGrid& grid, const CsrMatrix& a,
                                         const CsrMatrix& b) : grid_(grid), a_(expose(a)), b_(expose(b)) {}

GetInterNodeExchange::~GetInterNodeExchange() {
    // Abort before callers' input vectors are destroyed during exception unwinding.
    if (!closed_) fail("GET backend destroyed before collective close", grid_.world);
}

GetInterNodeExchange::InputWindow GetInterNodeExchange::expose(const CsrMatrix& input) {
    InputWindow result{{input.rows, input.cols, mpiCount(input.values.size())},
                       {input.rowPtr.data(), input.columnIndices.data(), input.values.data()}};
    checkInput(input, result);
    // With only one node every inter-node source is local, including singleton MPI jobs.
    if (grid_.side > 1) {
        result.windows[0] = exposeBuffer(input.rowPtr, grid_.world);
        result.windows[1] = exposeBuffer(input.columnIndices, grid_.world);
        result.windows[2] = exposeBuffer(input.values, grid_.world);
    }
    return result;
}

void GetInterNodeExchange::checkInput(const CsrMatrix& input, const InputWindow& exposed) const {
    if (input.rows < 0 || input.cols < 0 || input.rows != exposed.shape.rows ||
        input.cols != exposed.shape.cols || input.rowPtr.size() != static_cast<std::size_t>(input.rows) + 1 ||
        input.columnIndices.size() != input.values.size() || mpiCount(input.values.size()) != exposed.shape.nnz ||
        input.rowPtr.data() != exposed.bases[0] || input.columnIndices.data() != exposed.bases[1] ||
        input.values.data() != exposed.bases[2]) {
        throw std::invalid_argument("GET inputs must retain their exposed shapes, sizes and buffer addresses");
    }
}

void GetInterNodeExchange::syncInputs() {
    for (const auto* input : {&a_, &b_}) {
        for (const auto window : input->windows) {
            checkMpi(MPI_Win_sync(window), "MPI_Win_sync(GET input)", grid_.world);
        }
    }
}

void GetInterNodeExchange::begin(const CsrMatrix& a, const CsrMatrix& b) {
    if (active_ || closed_) throw std::logic_error("invalid GET product lifecycle");
    checkInput(a, a_);
    checkInput(b, b_);
    if (grid_.side > 1) {
        syncInputs();
        // Publish every rank's current inputs before any rank can read them remotely.
        checkMpi(MPI_Barrier(grid_.world), "MPI_Barrier(GET inputs ready)", grid_.world);
    }
    active_ = true;
}

void GetInterNodeExchange::getTile(CsrMatrix& tile, int source, const InputWindow& exposed) {
    checkMpi(MPI_Get(tile.rowPtr.data(), mpiCount(tile.rowPtr.size()), MPI_INT, source, 0,
                     mpiCount(tile.rowPtr.size()), MPI_INT, exposed.windows[0]),
             "MPI_Get(tile pointers)", grid_.world);
    const int nnz = mpiCount(tile.values.size());
    if (nnz > 0) {
        checkMpi(MPI_Get(tile.columnIndices.data(), nnz, MPI_INT, source, 0, nnz, MPI_INT,
                         exposed.windows[1]), "MPI_Get(tile indices)", grid_.world);
        checkMpi(MPI_Get(tile.values.data(), nnz, MPI_DOUBLE, source, 0, nnz, MPI_DOUBLE,
                         exposed.windows[2]), "MPI_Get(tile values)", grid_.world);
    }
}

void GetInterNodeExchange::complete(int source, const InputWindow& exposed) {
    for (const auto window : exposed.windows) {
        checkMpi(MPI_Win_flush(source, window), "MPI_Win_flush(GET tile)", grid_.world);
    }
}

void GetInterNodeExchange::fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                                ProductWorkspace& workspace) {
    if (!active_) throw std::logic_error("GET fetch outside a product");
    resizeTile(workspace.a, stage.a);
    resizeTile(workspace.b, stage.b);
    if (stage.aSource == grid_.rank) workspace.a = a;
    else getTile(workspace.a, stage.aSource, a_);
    if (stage.bSource == grid_.rank) workspace.b = b;
    else getTile(workspace.b, stage.bSource, b_);
    // Finish the entire current stage before intra-node aggregation or local computation.
    if (stage.aSource != grid_.rank) complete(stage.aSource, a_);
    if (stage.bSource != grid_.rank) complete(stage.bSource, b_);
}

void GetInterNodeExchange::finish() {
    if (!active_) throw std::logic_error("GET finish outside a product");
    if (grid_.side > 1) {
        // Each fetch flushed its reads. No exposed input may change until all readers finish.
        checkMpi(MPI_Barrier(grid_.world), "MPI_Barrier(GET readers done)", grid_.world);
        syncInputs();
    }
    active_ = false;
}

void GetInterNodeExchange::close() {
    if (active_ || closed_) throw std::logic_error("invalid GET close");
    for (auto* input : {&a_, &b_}) {
        for (auto& window : input->windows) {
            if (window == MPI_WIN_NULL) continue;
            checkMpi(MPI_Win_unlock_all(window), "MPI_Win_unlock_all(GET input)", grid_.world);
            checkMpi(MPI_Win_free(&window), "MPI_Win_free(GET input)", grid_.world);
        }
    }
    closed_ = true;
}

}  // namespace trident
