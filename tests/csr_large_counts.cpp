#include "spgemm_common.hpp"
#include <iostream>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("large-count arithmetic regression");
}
template <typename F> void rejects(F operation) {
    try { operation(); }
    catch (const std::overflow_error&) { return; }
    throw std::runtime_error("unrepresentable count accepted");
}
}

int main() {
    try {
        const CsrOffset boundary = std::numeric_limits<int>::max();
        require(checkedCsrCount(static_cast<std::size_t>(boundary + 1)) == boundary + 1);
        // Two valid local counts can produce a global prefix above INT_MAX.
        const auto global = checkedCsrAdd(1'500'000'000, 1'500'000'000);
        require(global == 3'000'000'000LL);
        require(checkedCsrAdd(std::numeric_limits<CsrOffset>::max() - 7, 7) ==
                std::numeric_limits<CsrOffset>::max());
        const std::vector<CsrOffset> pointers{0, boundary - 2, boundary + 7, global};
        require(pointers[2] - pointers[1] == 9 && pointers.back() == global);
        require(csrSize<double>(global) == 3'000'000'000ULL);
        rejects([&] { checkedMatrixDimension(boundary + 1); });
        rejects([] { checkedCsrAdd(std::numeric_limits<CsrOffset>::max(), 1); });
        rejects([] { checkedCsrAdd(-1, 1); });
        rejects([] { csrSize<double>(-1); });
        rejects([] { csrSize<double>(std::numeric_limits<CsrOffset>::max()); });
        rejects([] { csrWindowBytes<double>(std::numeric_limits<std::size_t>::max()); });
        require(csrWindowBytes<CsrOffset>(5) == 40);
        require(csrMpiDisplacement(boundary + 17) == boundary + 17);

        for (const CsrOffset count : {CsrOffset{0}, boundary - 1, boundary, boundary + 1,
                                     boundary * 2, boundary * 2 + 9}) {
            CsrOffset visited = 0;
            forMpiChunks(count, [&](CsrOffset offset, int chunk) {
                require(offset == visited && chunk == std::min<CsrOffset>(boundary, count - offset));
                visited += chunk;
            }, boundary);
            require(visited == count);
        }
        for (const CsrOffset count : {0, 1, 6, 7, 8, 14, 15}) {
            CsrOffset visited = 0;
            forMpiChunks(count, [&](CsrOffset offset, int chunk) {
                require(offset == visited && chunk == std::min<CsrOffset>(7, count - offset));
                visited += chunk;
            }, 7);
            require(visited == count);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
