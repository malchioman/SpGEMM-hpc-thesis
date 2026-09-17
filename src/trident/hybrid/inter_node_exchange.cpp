#include "inter_node_exchange.hpp"

#include "csr_transfer.hpp"

#include <limits>
#include <stdexcept>

namespace trident {

HybridInterNodeExchange::HybridInterNodeExchange(const ProcessGrid& grid) : grid_(grid) {
    int provided = MPI_THREAD_SINGLE;
    checkMpi(MPI_Query_thread(&provided), "MPI_Query_thread", grid_.world);
    if (provided < MPI_THREAD_MULTIPLE) {
        throw std::runtime_error("Trident hybrid requires MPI_THREAD_MULTIPLE");
    }
    checkMpi(MPI_Comm_dup(grid_.world, &payload_), "MPI_Comm_dup(hybrid)", grid_.world);
    const int slots = 2 * grid_.side;
    void* storage = nullptr;
    checkMpi(MPI_Win_allocate(static_cast<MPI_Aint>(slots) * sizeof(std::uint64_t),
                             sizeof(std::uint64_t), MPI_INFO_NULL, payload_, &storage, &requests_),
             "MPI_Win_allocate(requests)", grid_.world);
    checkMpi(MPI_Win_lock_all(0, requests_), "MPI_Win_lock_all(requests)", grid_.world);
    const std::vector<std::uint64_t> zeros(slots, 0);
    // Even self accesses use RMA atomics; no concurrent native loads/stores of window memory.
    checkMpi(MPI_Accumulate(zeros.data(), slots, MPI_UINT64_T, grid_.rank, 0, slots,
                           MPI_UINT64_T, MPI_REPLACE, requests_), "MPI_Accumulate(init)", grid_.world);
    checkMpi(MPI_Win_flush(grid_.rank, requests_), "MPI_Win_flush(init)", grid_.world);
    checkMpi(MPI_Barrier(payload_), "MPI_Barrier(requests ready)", grid_.world);
}

HybridInterNodeExchange::~HybridInterNodeExchange() {
    if (service_.joinable()) {
        fail("hybrid product interrupted with an active service thread", grid_.world);
    }
}

void HybridInterNodeExchange::begin(const CsrMatrix& a, const CsrMatrix& b) {
    if (active_ || requests_ == MPI_WIN_NULL) throw std::logic_error("invalid hybrid product lifecycle");
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("hybrid request generation exhausted");
    }
    ++generation_;
    active_ = true;
    if (grid_.side > 1) {
        service_ = std::thread([this, &a, &b, generation = generation_] {
            try {
                serve(a, b, generation);
            } catch (const std::exception& error) {
                fail(error.what(), grid_.world);
            } catch (...) {
                fail("unknown hybrid service error", grid_.world);
            }
        });
    }
}

void HybridInterNodeExchange::notify(int target, int slot) {
    checkMpi(MPI_Accumulate(&generation_, 1, MPI_UINT64_T, target, slot, 1, MPI_UINT64_T,
                           MPI_REPLACE, requests_), "MPI_Accumulate(request)", grid_.world);
    checkMpi(MPI_Win_flush(target, requests_), "MPI_Win_flush(request)", grid_.world);
    ++published_;
}

void HybridInterNodeExchange::fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                                   ProductWorkspace& workspace) {
    if (!active_) throw std::logic_error("hybrid fetch outside a product");
    resizeTile(workspace.a, stage.a);
    resizeTile(workspace.b, stage.b);
    std::vector<MPI_Request> receives;
    receives.reserve(6);
    if (stage.aSource == grid_.rank) workspace.a = a;
    else receiveTile(workspace.a, stage.aSource, 20, payload_, receives);
    if (stage.bSource == grid_.rank) workspace.b = b;
    else receiveTile(workspace.b, stage.bSource, 30, payload_, receives);
    // Receives must be posted before publishing requests, including rendezvous-size payloads.
    if (stage.aSource != grid_.rank) notify(stage.aSource, grid_.col);
    if (stage.bSource != grid_.rank) notify(stage.bSource, grid_.side + grid_.row);
    waitAll(receives, grid_.world);
}

void HybridInterNodeExchange::serve(const CsrMatrix& a, const CsrMatrix& b, std::uint64_t generation) {
    std::vector<bool> consumed(2 * grid_.side, false);
    consumed[grid_.col] = true;
    consumed[grid_.side + grid_.row] = true;
    int remaining = 2 * (grid_.side - 1);
    std::vector<MPI_Request> sends;
    sends.reserve(3);
    while (remaining > 0) {
        bool progressed = false;
        for (int slot = 0; slot < 2 * grid_.side; ++slot) {
            if (consumed[slot]) continue;
            std::uint64_t observed = 0;
            checkMpi(MPI_Fetch_and_op(nullptr, &observed, MPI_UINT64_T, grid_.rank, slot,
                                     MPI_NO_OP, requests_), "MPI_Fetch_and_op(poll)", grid_.world);
            checkMpi(MPI_Win_flush(grid_.rank, requests_), "MPI_Win_flush(poll)", grid_.world);
            if (observed < generation) continue;
            if (observed != generation) throw std::runtime_error("hybrid request generation mismatch");
            const bool forA = slot < grid_.side;
            const int target = forA ? grid_.rankAt(grid_.row, slot, grid_.nodeRank)
                                   : grid_.rankAt(slot - grid_.side, grid_.col, grid_.nodeRank);
            sends.clear();
            sendTile(forA ? a : b, target, forA ? 20 : 30, payload_, sends);
            waitAll(sends, grid_.world);
            consumed[slot] = true;
            --remaining;
            ++served_;
            progressed = true;
        }
        if (!progressed) std::this_thread::yield();
    }
}

void HybridInterNodeExchange::finish() {
    if (!active_) throw std::logic_error("hybrid finish outside a product");
    // Every remote consumer requests each owned input once per product. Joining drains all
    // responses before input buffers may change; generation slots never need a racy reset.
    if (service_.joinable()) service_.join();
    active_ = false;
}

void HybridInterNodeExchange::close() {
    if (active_) throw std::logic_error("cannot close an active hybrid product");
    checkMpi(MPI_Win_unlock_all(requests_), "MPI_Win_unlock_all(requests)", grid_.world);
    checkMpi(MPI_Win_free(&requests_), "MPI_Win_free(requests)", grid_.world);
    checkMpi(MPI_Comm_free(&payload_), "MPI_Comm_free(hybrid)", grid_.world);
}

}  // namespace trident
