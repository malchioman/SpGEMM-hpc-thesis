#pragma once

#include "csr_matrix.hpp"
#include <mpi.h>
#include <algorithm>
#include <type_traits>

void checkMpi(int error, const char* call, MPI_Comm communicator);

// One build-wide limit: both endpoints must use the same partition of a payload.
// A small value exercises the production transfer paths in regression tests.
#ifndef SPGEMM_MPI_CHUNK_LIMIT
#define SPGEMM_MPI_CHUNK_LIMIT 2147483647
#endif
inline constexpr CsrOffset mpiChunkLimit = SPGEMM_MPI_CHUNK_LIMIT;
static_assert(mpiChunkLimit > 0 && mpiChunkLimit <= std::numeric_limits<int>::max());

template <typename T> MPI_Datatype csrMpiType() {
    if constexpr (std::is_same_v<T, int>) return MPI_INT;
    else if constexpr (std::is_same_v<T, CsrOffset>) return MPI_INT64_T;
    else {
        static_assert(std::is_same_v<T, double>, "unsupported CSR MPI element type");
        return MPI_DOUBLE;
    }
}

template <typename F>
void forMpiChunks(CsrOffset count, F&& operation, CsrOffset limit = mpiChunkLimit) {
    if (count < 0 || limit <= 0 || limit > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid MPI payload count/chunk limit");
    for (CsrOffset offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(limit, count - offset));
        operation(offset, chunk);
        offset += chunk;  // Bounded by count, including at INT64_MAX.
    }
}

inline MPI_Aint csrMpiDisplacement(CsrOffset offset) {
    if (offset < 0 || static_cast<std::uint64_t>(offset) >
                        static_cast<std::uint64_t>(std::numeric_limits<MPI_Aint>::max()))
        throw std::overflow_error("CSR RMA displacement exceeds MPI_Aint");
    return static_cast<MPI_Aint>(offset);
}

template <typename T> MPI_Aint csrWindowBytes(std::size_t count) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(T))
        throw std::overflow_error("CSR RMA window bytes exceed MPI_Aint");
    return static_cast<MPI_Aint>(count) * static_cast<MPI_Aint>(sizeof(T));
}

template <typename T>
void csrSend(const T* data, CsrOffset count, int peer, int tag, MPI_Comm comm) {
    csrSize<T>(count);
    forMpiChunks(count, [&](CsrOffset offset, int chunk) {
        checkMpi(MPI_Send(data + offset, chunk, csrMpiType<T>(), peer, tag, comm), "MPI_Send(CSR chunk)", comm);
    });
}
template <typename T>
void csrRecv(T* data, CsrOffset count, int peer, int tag, MPI_Comm comm) {
    csrSize<T>(count);
    forMpiChunks(count, [&](CsrOffset offset, int chunk) {
        checkMpi(MPI_Recv(data + offset, chunk, csrMpiType<T>(), peer, tag, comm, MPI_STATUS_IGNORE),
                 "MPI_Recv(CSR chunk)", comm);
    });
}
template <typename T>
void csrIsend(const T* data, CsrOffset count, int peer, int tag, MPI_Comm comm,
              std::vector<MPI_Request>& requests) {
    csrSize<T>(count);
    forMpiChunks(count, [&](CsrOffset offset, int chunk) {
        requests.push_back(MPI_REQUEST_NULL);
        checkMpi(MPI_Isend(data + offset, chunk, csrMpiType<T>(), peer, tag, comm, &requests.back()),
                 "MPI_Isend(CSR chunk)", comm);
    });
}
template <typename T>
void csrIrecv(T* data, CsrOffset count, int peer, int tag, MPI_Comm comm,
              std::vector<MPI_Request>& requests) {
    csrSize<T>(count);
    forMpiChunks(count, [&](CsrOffset offset, int chunk) {
        requests.push_back(MPI_REQUEST_NULL);
        checkMpi(MPI_Irecv(data + offset, chunk, csrMpiType<T>(), peer, tag, comm, &requests.back()),
                 "MPI_Irecv(CSR chunk)", comm);
    });
}
template <typename T>
void csrGet(T* data, CsrOffset count, int peer, CsrOffset displacement, MPI_Win window, MPI_Comm comm) {
    csrSize<T>(count);
    csrMpiDisplacement(checkedCsrAdd(displacement, count, "GET target range"));
    forMpiChunks(count, [&](CsrOffset offset, int chunk) {
        checkMpi(MPI_Get(data + offset, chunk, csrMpiType<T>(), peer,
                         csrMpiDisplacement(displacement + offset), chunk, csrMpiType<T>(), window),
                 "MPI_Get(CSR chunk)", comm);
    });
}
template <typename T>
void csrPut(const T* data, CsrOffset count, int peer, CsrOffset displacement, MPI_Win window, MPI_Comm comm) {
    csrSize<T>(count);
    csrMpiDisplacement(checkedCsrAdd(displacement, count, "PUT target range"));
    forMpiChunks(count, [&](CsrOffset offset, int chunk) {
        checkMpi(MPI_Put(data + offset, chunk, csrMpiType<T>(), peer,
                         csrMpiDisplacement(displacement + offset), chunk, csrMpiType<T>(), window),
                 "MPI_Put(CSR chunk)", comm);
    });
}
