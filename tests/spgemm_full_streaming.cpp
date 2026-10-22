#include "spgemm_common.hpp"
#include "matrix_market.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void equivalent(const CsrMatrix& a, const CsrMatrix& b, const CsrMatrix& c) {
    const auto expected = maxAbsoluteDifference(c, serialSpgemm(a, b));
    const auto actual = validateProduct(a, b, c, ValidationMode::Full);
    require(actual.maxAbsoluteError && *actual.maxAbsoluteError == expected,
            "streaming error differs from materialized Full oracle");
}
CsrMatrix input(int rows, int cols, std::mt19937& random) {
    CsrMatrix a{rows, cols, {0}, {}, {}};
    for (int r = 0; r < rows; ++r) {
        // Deliberately preserve shuffled duplicate inputs, cancellation and explicit zeros.
        for (int k = 0; k < 2 * cols; ++k) {
            a.columnIndices.push_back(static_cast<int>(random() % cols));
            a.values.push_back(std::ldexp(static_cast<double>(int(random() % 9) - 4), -20));
        }
        a.rowPtr.push_back(checkedCsrCount(a.values.size()));
    }
    return a;
}
}

int main() {
    try {
        std::mt19937 random(42);
        for (int i = 0; i < 32; ++i) {
            const auto a = input(i % 7 + 1, 5, random);
            const auto b = input(5, i % 4 + 1, random);
            auto c = serialSpgemm(a, b);
            equivalent(a, b, c);
            require(validateProduct(a, b, c, ValidationMode::Full).passed, "exact dyadic product rejected");
            if (!c.values.empty()) {
                c.values[0] += 1;
                equivalent(a, b, c);
                require(!validateProduct(a, b, c, ValidationMode::Full).passed,
                        "wrong value accepted despite unchanged row counts");
            }
            const CsrMatrix missing{a.rows, b.cols, std::vector<CsrOffset>(a.rows + 1, 0), {}, {}};
            equivalent(a, b, missing);
        }
        const double tiny = std::ldexp(1.0, -55);
        const CsrMatrix a{1, 3, {0, 3}, {0, 1, 2}, {1, 1, 1}};
        const CsrMatrix b{3, 1, {0, 1, 2, 3}, {0, 0, 0}, {1, tiny, -1}};
        // Serial 1+tiny-1 is zero; reordered -1+1+tiny retains a tiny residual.
        const CsrMatrix reordered{1, 1, {0, 1}, {0}, {tiny}};
        require(!validateProduct(a, b, reordered, ValidationMode::Partial).passed,
                "strict partial control should still reject different counts");
        const auto accepted = validateProduct(a, b, reordered, ValidationMode::Full);
        require(accepted.passed && *accepted.maxAbsoluteError == tiny && accepted.rowNnzMismatches == 1,
                "numerically valid sparse residual should pass Full");
        equivalent(a, b, reordered);

        const CsrMatrix unit{1, 1, {0, 1}, {0}, {1}};
        for (double value : {0.5e-10, 1e-10, 1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
            auto changed = reordered;
            changed.values[0] = value;
            equivalent(a, b, changed);
            require(!validateProduct(a, b, changed, ValidationMode::Full).passed,
                    "corruption greater than the dot-product error bound accepted");
        }
        const CsrMatrix colA{1, 1, {0, 1}, {0}, {1}};
        const CsrMatrix colB{1, 2, {0, 1}, {0}, {1}};
        const CsrMatrix wrongCol{1, 2, {0, 1}, {1}, {1}};
        equivalent(colA, colB, wrongCol);
        require(!validateProduct(colA, colB, wrongCol, ValidationMode::Full).passed, "wrong column accepted");
        for (double value : {1e200, 1e-200, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
            const CsrMatrix large{1, 1, {0, 1}, {0}, {value}};
            equivalent(large, large, unit);
        }
        const CsrMatrix zeroRows{0, 1, {0}, {}, {}};
        equivalent(zeroRows, unit, zeroRows);
        equivalent(unit, unit, {2, 1, {0, 0, 0}, {}, {}});
        for (const auto& malformed : {
                CsrMatrix{1, 1, {0, 2}, {0}, {1}},
                CsrMatrix{1, 1, {0, 2}, {0, 0}, {0.5, 0.5}},
                CsrMatrix{1, 1, {0, 1}, {1}, {1}}})
            require(!validateProduct(unit, unit, malformed, ValidationMode::Full).passed,
                    "malformed result CSR accepted");
        // Scale must matter: an absolute error much smaller than 1e-10 can still be a bug.
        const CsrMatrix small{1, 1, {0, 1}, {0}, {1e-150}};
        const CsrMatrix corruptedSmall{1, 1, {0, 1}, {0}, {1e-300 + 1e-310}};
        require(!validateProduct(small, small, corruptedSmall, ValidationMode::Full).passed,
                "small-scale corruption hidden by a global absolute tolerance");
        require(validateProduct(small, small, serialSpgemm(small, small), ValidationMode::Full).passed,
                "small nonzero result rejected");
        const CsrMatrix underflow{1, 1, {0, 1}, {0}, {1e-200}};
        require(validateProduct(underflow, underflow, {1, 1, {0, 0}, {}, {}}, ValidationMode::Full).passed,
                "valid gradual underflow rejected");
        const CsrMatrix subnormal{1, 1, {0, 1}, {0}, {std::numeric_limits<double>::denorm_min()}};
        require(validateProduct(subnormal, unit, subnormal, ValidationMode::Full).passed,
                "subnormal operands rejected");
        // No scalar products contribute to column 1: no rounding budget may hide an extra value.
        const CsrMatrix extraTiny{1, 2, {0, 2}, {0, 1}, {1, 1e-300}};
        require(!validateProduct(colA, colB, extraTiny, ValidationMode::Full).passed,
                "spurious column without contributing terms accepted");
        for (const auto& badInput : {
                CsrMatrix{1, 1, {0, 1}, {-1}, {1}},
                CsrMatrix{1, 1, {0, 1}, {1}, {1}},
                CsrMatrix{1, 1, {0, 1}, {0}, {std::numeric_limits<double>::quiet_NaN()}}}) {
            require(!validateProduct(badInput, unit, unit, ValidationMode::Full).passed,
                    "invalid A accepted");
            require(!validateProduct(unit, badInput, unit, ValidationMode::Full).passed,
                    "invalid B accepted");
        }
        // The model cannot silently accept a different rounding mode.
        require(std::fesetround(FE_DOWNWARD) == 0, "cannot set test rounding mode");
        const bool badEnvironment = validateProduct(unit, unit, unit, ValidationMode::Full).passed;
        require(std::fesetround(FE_TONEAREST) == 0, "cannot restore rounding mode");
        require(!badEnvironment, "unsupported rounding mode accepted");
#if defined(__SSE2__)
        const unsigned originalControl = _mm_getcsr();
        for (const unsigned flag : {0x8000u, 0x0040u}) { // FTZ and DAZ independently.
            _mm_setcsr(originalControl | flag);
            const bool acceptedEnvironment = validateProduct(unit, unit, unit, ValidationMode::Full).passed;
            _mm_setcsr(originalControl);
            require(!acceptedEnvironment, "flush-to-zero or denormals-as-zero accepted");
        }
#endif

        // Real HV15R witnesses: preserve dimensions, operands, and the numerical results
        // independently established with exact rational arithmetic and actual MPI binaries.
        const auto fixtures = std::filesystem::path(__FILE__).parent_path() / "data/hv15r_roundoff";
        for (const auto& name : {"zero-cancellation-witness", "absolute-threshold-witness"}) {
            const auto realA = readMatrixMarket((fixtures / (std::string(name) + "-A.mtx")).string());
            const auto realB = readMatrixMarket((fixtures / (std::string(name) + "-B.mtx")).string());
            const bool zero = std::string(name) == "zero-cancellation-witness";
            const int row = zero ? 426610 : 45207, column = zero ? 448836 : 1397793;
            CsrMatrix candidate{realA.rows, realB.cols,
                                std::vector<CsrOffset>(static_cast<std::size_t>(realA.rows) + 1, 0), {}, {}};
            if (!zero) {
                candidate.columnIndices.push_back(column);
                candidate.values.push_back(73368.519419447723);
                std::fill(candidate.rowPtr.begin() + row + 1, candidate.rowPtr.end(), 1);
            }
            const auto valid = validateProduct(realA, realB, candidate, ValidationMode::Full);
            require(valid.passed, "real roundoff witness rejected");
            require(*valid.maxAbsoluteError == (zero ? 1.8135240096338719e-20 : 1.0186340659856796e-10),
                    "historical TSV absolute error changed");
            if (zero) {
                candidate.columnIndices.push_back(column);
                candidate.values.push_back(0);
                std::fill(candidate.rowPtr.begin() + row + 1, candidate.rowPtr.end(), 1);
            }
            candidate.values[0] += 0.001;
            require(!validateProduct(realA, realB, candidate, ValidationMode::Full).passed,
                    "corrupted real HV15R coordinate accepted");
        }
        std::cout << "Full streaming: roundoff bounds, real witnesses, corruption rejection: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
