#include "spgemm_common.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Action> void rejects(Action action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected rejection");
}
// Frozen pre-cache Partial criterion, independent of the new builder/codec/comparator.
ValidationResult legacy(const CsrMatrix& a, const CsrMatrix& b, const CsrMatrix& c) {
    ValidationResult result;
    result.mode = ValidationMode::Partial;
    if (c.rows != a.rows || c.cols != b.cols) { result.passed = false; return result; }
    for (double value : c.values) if (!std::isfinite(value)) result.passed = false;
    std::unordered_map<int, double> accumulator;
    for (int row = 0; row < a.rows; ++row) {
        accumulator.clear();
        for (CsrOffset i = a.rowPtr[row]; i < a.rowPtr[row + 1]; ++i)
            for (CsrOffset j = b.rowPtr[a.columnIndices[i]]; j < b.rowPtr[a.columnIndices[i] + 1]; ++j)
                accumulator[b.columnIndices[j]] += a.values[i] * b.values[j];
        CsrOffset count = 0;
        for (const auto& entry : accumulator) {
            if (!std::isfinite(entry.second)) result.passed = false;
            if (entry.second != 0.0) ++count;
        }
        if (count != c.rowPtr[row + 1] - c.rowPtr[row]) ++result.rowNnzMismatches;
    }
    result.passed = result.passed && result.rowNnzMismatches == 0;
    return result;
}
void equivalent(const CsrMatrix& a, const CsrMatrix& b, const CsrMatrix& c) {
    const auto expected = legacy(a, b, c);
    const auto key = partialReferenceKey(a, b);
    const auto reference = decodePartialReference(encodePartialReference(buildPartialReference(a, b), key),
                                                   key, a.rows, b.cols);
    for (const auto actual : {comparePartialReference(reference, c),
                              validateProduct(a, b, c, ValidationMode::Partial)}) {
        require(expected.passed == actual.passed && expected.rowNnzMismatches == actual.rowNnzMismatches &&
                !actual.maxAbsoluteError, "Partial criterion changed");
    }
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "test requires a scratch directory");
        const std::filesystem::path directory = argv[1];
        std::filesystem::create_directories(directory);
        const CsrMatrix a{3, 2, {0, 2, 2, 3}, {0, 1, 1}, {1, 1, 2}};
        const CsrMatrix b{2, 4, {0, 2, 4}, {0, 1, 0, 2}, {1, 2, -1, 3}};
        const auto product = serialSpgemm(a, b);
        equivalent(a, b, product);
        require(buildPartialReference(a, b).rowNnz == std::vector<CsrOffset>({2, 0, 2}), "exact cancellation lost");
        auto wrong = product;
        wrong.rowPtr = {0, 1, 2, 4}; equivalent(a, b, wrong);
        wrong = product; wrong.cols = 9; equivalent(a, b, wrong);
        wrong = product; wrong.rows = 1; equivalent(a, b, wrong);
        wrong = product; wrong.values[0] = 999; wrong.columnIndices[0] = 0;
        equivalent(a, b, wrong);  // Deliberately still passes Partial.
        for (double v : {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
            wrong = product; wrong.values[0] = v; equivalent(a, b, wrong);
            auto altered = b; altered.values[0] = v; equivalent(a, altered, product);
        }
        const CsrMatrix empty{3, 2, {0, 0, 0, 0}, {}, {}};
        equivalent(empty, b, serialSpgemm(empty, b));
        const CsrMatrix zeroRows{0, 2, {0}, {}, {}};
        equivalent(zeroRows, b, serialSpgemm(zeroRows, b));
        for (double value : {1.0, 1e200, 1e-200}) {
            const CsrMatrix square{1, 1, {0, 1}, {0}, {value}};
            equivalent(square, square, serialSpgemm(square, square));
        }
        // Non-associative sums and duplicate entries retain their input order.
        const CsrMatrix orderA{1, 1, {0, 1}, {0}, {1}};
        CsrMatrix orderB{1, 1, {0, 3}, {0, 0, 0}, {1e16, -1e16, 1}};
        require(buildPartialReference(orderA, orderB).rowNnz[0] == 1, "wrong sum order");
        const auto beforeOrder = partialReferenceKey(orderA, orderB);
        orderB.values = {1e16, 1, -1e16};
        require(buildPartialReference(orderA, orderB).rowNnz[0] == 0 &&
                beforeOrder != partialReferenceKey(orderA, orderB), "order not part of identity");
        const auto key = partialReferenceKey(a, b);
        const PartialReference wide{2, 1, true, {CsrOffset(1) << 33, std::numeric_limits<CsrOffset>::max()}};
        const auto bytes = encodePartialReference(wide, key);
        require(decodePartialReference(bytes, key, 2, 1).rowNnz == wide.rowNnz, "64-bit codec narrowed counts");
        const PartialReference wideCompare{1, 1, true, {CsrOffset(1) << 33}};
        const CsrMatrix countOnly{1, 1, {0, CsrOffset(1) << 33}, {}, {}};
        require(comparePartialReference(wideCompare, countOnly).passed, "64-bit comparison narrowed counts");
        auto corrupt = bytes; corrupt[100] ^= 1;
        rejects([&] { decodePartialReference(corrupt, key, 2, 1); });
        corrupt = bytes; corrupt[7] = '2';
        rejects([&] { decodePartialReference(corrupt, key, 2, 1); });
        rejects([&] { decodePartialReference(bytes.substr(0, bytes.size()-1), key, 2, 1); });
        rejects([&] { decodePartialReference(bytes + "x", key, 2, 1); });
        rejects([&] { decodePartialReference(bytes, std::string(64, '0'), 2, 1); });
        rejects([&] { decodePartialReference(bytes, key, 3, 1); });
        ValidationDiagnostics diagnostics;
        preparePartialReference(a, b, directory, diagnostics);
        require(validateProduct(a, b, product, ValidationMode::Partial, directory.string()).passed, "cache failed");
        wrong = product; wrong.values[0] = std::numeric_limits<double>::infinity();
        require(!validateProduct(a, b, wrong, ValidationMode::Partial, directory.string()).passed, "reused PASS");
        auto changed = a; changed.values[0] += 1;
        require(partialReferenceKey(changed, b) != key, "A content not hashed");
        rejects([&] { validateProduct(changed, b, product, ValidationMode::Partial, directory.string()); });
        auto changedB = b; changedB.values[0] += 1;
        require(partialReferenceKey(a, changedB) != key, "B content not hashed");
        rejects([&] { validateProduct(a, changedB, product, ValidationMode::Partial, directory.string()); });
        const CsrMatrix huge{1, 1, {0, 1}, {0}, {1e200}};
        preparePartialReference(huge, huge, directory, diagnostics);
        require(!validateProduct(huge, huge, {1, 1, {0, 1}, {0}, {1}}, ValidationMode::Partial,
                                 directory.string()).passed, "non-finite reference passed");
        // Full and None ignore a supplied cache at the API level; CLI forbids accidental combinations.
        require(validateProduct(a, b, product, ValidationMode::Full, "nonexistent").passed, "Full changed");
        require(validateProduct(a, b, wrong, ValidationMode::None, "nonexistent").passed, "None changed");
        std::cout << "partial_reference_test: PASS\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
