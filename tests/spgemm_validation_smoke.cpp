#include "spgemm_common.hpp"

#include <array>
#include <iostream>
#include <limits>

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
    passed = expectError(finite, finite, 0.0, "equal finite matrices") && passed;
    passed = expectError(finite, differentValue, 1.0, "finite mismatch") && passed;
    passed = expectError(finite, differentColumn, 2.0, "different sparse columns") && passed;
    passed = expectError(empty, empty, 0.0, "empty matrices") && passed;
    passed = expectError(finite, empty, 1.0, "entry missing from reference") && passed;
    passed = expectError(empty, finite, 1.0, "entry missing from result") && passed;
    passed = expectError(finite, differentShape, infinity, "incompatible shapes") && passed;

    for (const double value : std::array<double, 3>{nan, infinity, -infinity}) {
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
