#include "full_reference.hpp"
#include "spgemm_common.hpp"

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifndef _WIN32
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected rejection");
}
void same(const ValidationResult& a, const ValidationResult& b) {
    require(a.passed == b.passed && a.maxAbsoluteError == b.maxAbsoluteError &&
            a.rowNnzMismatches == b.rowNnzMismatches && a.maxErrorRatio == b.maxErrorRatio &&
            a.rejectedCoordinates == b.rejectedCoordinates && a.referenceDigits == b.referenceDigits,
            "cached validation changed the numerical decision or diagnostics");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "scratch directory required");
        const auto directory = std::filesystem::path(argv[1]) / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(directory);
        const CsrMatrix a{3, 2, {0, 2, 2, 3}, {0, 1, 1}, {1, 1, 2}};
        const CsrMatrix b{2, 4, {0, 2, 4}, {0, 1, 0, 2}, {1, 2, -1, 3}};
        const auto c = serialSpgemm(a, b);
        ValidationDiagnostics d;
        prepareFullReference(a, b, directory, d);
        const auto key = fullReferenceKey(a, b);
        require(d.key == key, "wrong reference key");
        auto equivalent = [&](const CsrMatrix& candidate) {
            same(validateProduct(a, b, candidate, ValidationMode::Full),
                 validateProduct(a, b, candidate, ValidationMode::Full, "", directory.string()));
        };
        equivalent(c);
        auto wrong = c; wrong.values[0] += 1;
        equivalent(wrong);
        require(!validateProduct(a, b, wrong, ValidationMode::Full, "", directory.string()).passed,
                "cache reused an old PASS");
        wrong = c; wrong.columnIndices[0] = 0; equivalent(wrong);
        wrong = c; wrong.values[0] = std::numeric_limits<double>::quiet_NaN(); equivalent(wrong);
        wrong = c; wrong.cols++; equivalent(wrong);
        auto changed = a; changed.values[0] += 1;
        rejects([&] { validateProduct(changed, b, c, ValidationMode::Full, "", directory.string()); });
        require(key != fullReferenceKey(b, a), "pair order omitted from key");
        FullReferenceRow row;
        FullReferenceReader reader(directory, key, a.rows, b.cols);
        reader.readRow(row);
        require(row.size() == 3 && row[0].serial == 0, "cancelled coordinate lost");
        reader.readRow(row); require(row.empty(), "empty row lost");
        reader.readRow(row); reader.finish();
        rejects([&] { reader.readRow(row); });
        rejects([&] { FullReferenceReader other(directory, key, a.rows + 1, b.cols); });
        int builds = 0;
        d = {};
        prepareFullReferenceFile(directory, key, a.rows, b.cols,
                                 [&](int, FullReferenceRow&) { ++builds; }, d);
        require(builds == 0 && d.source == "reused" && d.buildSeconds == 0, "reference rebuilt on reuse");

        // Exact codec roundtrips, including long-double-only bits, subnormals and signed zero.
        const auto codec = directory / "codec";
        FullReferenceRow original;
        for (long double v : {0.0L, -0.0L, 1.0L, -1.0L, std::nextafter(1.0L, 2.0L),
                              std::numeric_limits<long double>::denorm_min(),
                              std::numeric_limits<long double>::min(), std::numeric_limits<long double>::max()})
            original.push_back({static_cast<int>(original.size()), 1.0, v, std::abs(v), CsrOffset(1) << 33});
        d = {};
        prepareFullReferenceFile(codec, key, 1, static_cast<int>(original.size()),
                                 [&](int, FullReferenceRow& out) { out = original; }, d);
        FullReferenceReader roundtrip(codec, key, 1, static_cast<int>(original.size()));
        roundtrip.readRow(row); roundtrip.finish();
        for (std::size_t i = 0; i < row.size(); ++i)
            require(row[i].reference == original[i].reference && row[i].bound == original[i].bound &&
                    std::signbit(row[i].reference) == std::signbit(original[i].reference) &&
                    row[i].terms == original[i].terms, "lossy Full reference codec");
        // Interruption never publishes a partial cache; retry truncates only its own .tmp.
        const auto interrupted = directory / "interrupted";
        rejects([&] { prepareFullReferenceFile(interrupted, key, 2, 8,
            [&](int i, FullReferenceRow& out) { if (i == 1) throw std::runtime_error("interrupted"); out = original; }, d); });
        require(!std::filesystem::exists(fullReferencePath(interrupted, key)), "partial cache published");
        prepareFullReferenceFile(interrupted, key, 2, 8, [&](int, FullReferenceRow& out) { out = original; }, d);

        const auto path = fullReferencePath(directory, key);
        std::ifstream input(path, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(input)), {}); input.close();
        auto write = [&](const std::string& data) {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(data.data(), static_cast<std::streamsize>(data.size()));
        };
        auto verify = [&] { validateProduct(a, b, c, ValidationMode::Full, "", directory.string()); };
        write(bytes.substr(0, bytes.size() - 1)); rejects(verify);
        write(bytes + "x"); rejects(verify);
        auto corrupt = bytes; corrupt[corrupt.size() - 2] ^= 1; write(corrupt); rejects(verify);
        corrupt = bytes; corrupt[140] ^= 1; write(corrupt); rejects(verify);
        rejects([&] { prepareFullReference(a, b, directory, d); }); // No silent repair/recompute.
        write(bytes); verify();
#ifndef _WIN32
        // Real process-owned lock: competitor fails, SIGKILL releases lock for retry.
        const auto locked = directory / "locked";
        int signals[2]; require(pipe(signals) == 0, "pipe failed");
        const auto child = fork(); require(child >= 0, "fork failed");
        if (child == 0) {
            close(signals[0]);
            prepareFullReferenceFile(locked, key, 1, 8, [&](int, FullReferenceRow& out) {
                out = original; const char ready = 'r'; if (::write(signals[1], &ready, 1) != 1) _exit(2);
                pause();
            }, d);
            _exit(1);
        }
        close(signals[1]); char ready;
        require(read(signals[0], &ready, 1) == 1, "child failed"); close(signals[0]);
        rejects([&] { prepareFullReferenceFile(locked, key, 1, 8, [&](int, FullReferenceRow& out) { out = original; }, d); });
        kill(child, SIGKILL); int status; waitpid(child, &status, 0);
        prepareFullReferenceFile(locked, key, 1, 8, [&](int, FullReferenceRow& out) { out = original; }, d);
#endif
        std::cout << "full_reference: PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
