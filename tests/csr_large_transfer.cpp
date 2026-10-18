#include "spgemm_common.hpp"
#include <algorithm>
#include <array>
#include <iostream>

namespace {
template <typename T> T value(int rank, std::size_t index) {
    if constexpr (std::is_same_v<T, CsrOffset>) {
        return static_cast<CsrOffset>(std::numeric_limits<int>::max()) + 100 + rank * 1000 +
               static_cast<CsrOffset>(index);
    } else return static_cast<T>(100 + rank * 1000 + index);
}
template <typename T> void check(const std::vector<T>& data, int rank) {
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] != value<T>(rank, i)) throw std::runtime_error("CSR transfer corrupted data");
}
template <typename T> void transfers(int rank, int count) {
    const int peer = 1 - rank;
    std::vector<T> source(count), received(count, T{});
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = value<T>(rank, i);
    if (rank == 0) csrSend(source.data(), count, peer, 100, MPI_COMM_WORLD);
    else { csrRecv(received.data(), count, peer, 100, MPI_COMM_WORLD); check(received, peer); }
    std::fill(received.begin(), received.end(), T{});
    std::vector<MPI_Request> requests;
    csrIrecv(received.data(), count, peer, 101, MPI_COMM_WORLD, requests);
    csrIsend(source.data(), count, peer, 101, MPI_COMM_WORLD, requests);
    waitAll(requests, MPI_COMM_WORLD);
    check(received, peer);

    // Offset and guard elements detect incorrect disp_unit, displacement and last-chunk size.
    constexpr int padding = 3;
    std::vector<T> exposed(static_cast<std::size_t>(count) + 2 * padding, T{-1});
    std::copy(source.begin(), source.end(), exposed.begin() + padding);
    MPI_Win window = MPI_WIN_NULL;
    checkMpi(MPI_Win_create(exposed.data(), csrWindowBytes<T>(exposed.size()), sizeof(T),
                           MPI_INFO_NULL, MPI_COMM_WORLD, &window), "create test window", MPI_COMM_WORLD);
    checkMpi(MPI_Win_lock_all(0, window), "lock test window", MPI_COMM_WORLD);
    checkMpi(MPI_Win_sync(window), "publish test window", MPI_COMM_WORLD);
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "test ready", MPI_COMM_WORLD);
    std::fill(received.begin(), received.end(), T{});
    csrGet(received.data(), count, peer, padding, window, MPI_COMM_WORLD);
    checkMpi(MPI_Win_flush(peer, window), "complete test GET", MPI_COMM_WORLD);
    check(received, peer);
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "test readers done", MPI_COMM_WORLD);
    csrPut(source.data(), count, peer, padding, window, MPI_COMM_WORLD);
    checkMpi(MPI_Win_flush(peer, window), "complete test PUT", MPI_COMM_WORLD);
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "test writers done", MPI_COMM_WORLD);
    checkMpi(MPI_Win_sync(window), "observe test PUT", MPI_COMM_WORLD);
    std::copy_n(exposed.begin() + padding, count, received.begin());
    check(received, peer);
    for (int i = 0; i < padding; ++i)
        if (exposed[i] != T{-1} || exposed[padding + count + i] != T{-1})
            throw std::runtime_error("CSR RMA wrote beyond its target range");
    checkMpi(MPI_Win_unlock_all(window), "unlock test window", MPI_COMM_WORLD);
    checkMpi(MPI_Win_free(&window), "free test window", MPI_COMM_WORLD);
}
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        if (size != 2) throw std::runtime_error("csr_large_transfer requires two ranks");
        for (const int count : {0, 1, 6, 7, 8, 14, 15, 37}) {
            transfers<int>(rank, count);
            transfers<CsrOffset>(rank, count);
            transfers<double>(rank, count);
        }
    } catch (const std::exception& error) { fail(error.what(), MPI_COMM_WORLD); }
    MPI_Finalize();
    return 0;
}
