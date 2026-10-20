#include "spgemm_common.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <string>

namespace {

bool expectError(const CsrMatrix& lhs, const CsrMatrix& rhs, double expected,
                 const char* description) {
    const double actual = maxAbsoluteDifference(lhs, rhs);
    if (actual != expected) {
        std::cerr << description << ": expected error=" << expected << ", got=" << actual << '\n';
        return false;
    }
    return true;
}

bool expectPartial(const CsrMatrix& a, const CsrMatrix& b, const CsrMatrix& result,
                   bool expected, int mismatches, const char* description) {
    const auto validation = validateProduct(a, b, result, ValidationMode::Partial);
    if (validation.passed != expected || validation.rowNnzMismatches != mismatches ||
        validation.maxAbsoluteError ||
        std::string(validation.status()) != (expected ? "PARTIAL_PASS" : "PARTIAL_FAIL")) {
        std::cerr << "partial validation: " << description << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const CsrMatrix finite{1, 2, {0, 1}, {0}, {1.0}};
    const CsrMatrix differentValue{1, 2, {0, 1}, {0}, {2.0}};
    const CsrMatrix differentColumn{1, 2, {0, 1}, {1}, {2.0}};
    const CsrMatrix empty{1, 2, {0, 0}, {}, {}};
    const CsrMatrix differentShape{2, 2, {0, 0, 0}, {}, {}};

    bool passed = true;
    // Rectangular product with an empty row and exact cancellation in row 0.
    const CsrMatrix a{3, 2, {0, 2, 2, 3}, {0, 1, 1}, {1, 1, 2}};
    const CsrMatrix b{2, 3, {0, 2, 4}, {0, 1, 0, 2}, {1, 2, -1, 3}};
    const CsrMatrix product{3, 3, {0, 2, 2, 4}, {1, 2, 0, 2}, {2, 3, -2, 6}};
    passed = expectPartial(a, b, product, true, 0, "cancellation and empty row") && passed;
    auto altered = product;
    altered.values[0] = 99;
    altered.columnIndices[0] = 0;
    passed = expectPartial(a, b, altered, true, 0, "values and columns are not compared") && passed;
    if (validateProduct(a, b, altered, ValidationMode::Full).passed) {
        std::cerr << "full validation must reject altered values and columns\n";
        passed = false;
    }
    altered = product;
    altered.rowPtr = {0, 1, 2, 4};
    passed = expectPartial(a, b, altered, false, 2, "same total nnz, different row counts") && passed;
    altered = product;
    altered.cols = 4;
    passed = expectPartial(a, b, altered, false, 0, "incompatible result columns") && passed;
    passed = expectPartial(a, b, empty, false, 0, "incompatible result rows") && passed;
    const CsrMatrix emptyA{3, 2, {0, 0, 0, 0}, {}, {}};
    const CsrMatrix emptyProduct{3, 3, {0, 0, 0, 0}, {}, {}};
    passed = expectPartial(emptyA, b, emptyProduct, true, 0, "empty product") && passed;
    passed = expectPartial(a, b, emptyProduct, false, 2, "missing nonempty rows") && passed;
    passed = expectPartial(emptyA, b, product, false, 2, "unexpected nonempty rows") && passed;
    const CsrMatrix zeroRows{0, 2, {0}, {}, {}};
    passed = expectPartial(zeroRows, b, {0, 3, {0}, {}, {}}, true, 0, "zero rows") && passed;
    const CsrMatrix huge{1, 1, {0, 1}, {0}, {1e200}};
    const CsrMatrix unit{1, 1, {0, 1}, {0}, {1}};
    passed = expectPartial(huge, huge, unit, false, 0, "overflow in serial reference") && passed;
    const CsrMatrix tiny{1, 1, {0, 1}, {0}, {1e-200}};
    passed = expectPartial(tiny, tiny, {1, 1, {0, 0}, {}, {}}, true, 0, "underflow to zero") && passed;
    passed = expectError(finite, finite, 0.0, "equal finite matrices") && passed;
    passed = expectError(finite, differentValue, 1.0, "finite mismatch") && passed;
    passed = expectError(finite, differentColumn, 2.0, "different sparse columns") && passed;
    passed = expectError(empty, empty, 0.0, "empty matrices") && passed;
    passed = expectError(finite, empty, 1.0, "entry missing from reference") && passed;
    passed = expectError(empty, finite, 1.0, "entry missing from result") && passed;
    passed = expectError(finite, differentShape, infinity, "incompatible shapes") && passed;

    for (const double value : std::array<double, 3>{nan, infinity, -infinity}) {
        altered = product;
        altered.values[0] = value;
        passed = expectPartial(a, b, altered, false, 0, "non-finite result") && passed;
        auto nonFiniteB = b;
        nonFiniteB.values[1] = value;
        passed = expectPartial(a, nonFiniteB, product, false, 0, "non-finite reference") && passed;
        const CsrMatrix nonFinite{1, 2, {0, 1}, {0}, {value}};
        passed = expectError(nonFinite, finite, infinity, "non-finite result") && passed;
        passed = expectError(finite, nonFinite, infinity, "non-finite reference") && passed;
        passed = expectError(nonFinite, nonFinite, infinity, "both non-finite") && passed;
        passed = expectError(nonFinite, empty, infinity, "unmatched non-finite result") && passed;
        passed = expectError(empty, nonFinite, infinity, "unmatched non-finite reference") && passed;
    }

    const CsrMatrix largePositive{1, 2, {0, 1}, {0}, {std::numeric_limits<double>::max()}};
    const CsrMatrix largeNegative{1, 2, {0, 1}, {0}, {-std::numeric_limits<double>::max()}};
    passed = expectError(largePositive, largeNegative, infinity, "difference overflow") && passed;

    for (const double error : std::array<double, 2>{0.0, 0.5e-10}) {
        if (!validationPassed(error)) {
            std::cerr << "unexpected validation failure for error=" << error << '\n';
            passed = false;
        }
    }
    for (const double error : std::array<double, 5>{1.0e-10, 1.0, infinity, -infinity, nan}) {
        if (validationPassed(error)) {
            std::cerr << "unexpected validation success for error=" << error << '\n';
            passed = false;
        }
    }
    return passed ? 0 : 1;
}
