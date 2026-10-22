#pragma once

#include "partial_reference.hpp"

#include <functional>
#include <memory>

// Includes cancelled coordinates. Values and bounds are stored losslessly;
// this is a reference, never the outcome of validating an earlier result.
struct FullReferenceCell {
    int column = 0;
    double serial = 0;
    long double reference = 0;
    long double bound = 0;
    CsrOffset terms = 0;
};
using FullReferenceRow = std::vector<FullReferenceCell>;

std::string fullReferenceKey(const CsrMatrix& a, const CsrMatrix& b);
std::filesystem::path fullReferencePath(const std::filesystem::path& directory, const std::string& key);

// Sequential, bounded-memory reader. finish() verifies the complete SHA-256
// before the caller may report PASS. Missing/invalid files never trigger computation.
class FullReferenceReader {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    FullReferenceReader(const std::filesystem::path& directory, const std::string& key, int rows, int cols);
    ~FullReferenceReader();
    void readRow(FullReferenceRow& row);
    void finish();
};

void prepareFullReferenceFile(const std::filesystem::path& directory, const std::string& key,
                             int rows, int cols, const std::function<void(int, FullReferenceRow&)>& buildRow,
                             ValidationDiagnostics& diagnostics);
// Numeric builder shared with uncached Full validation; standalone compute-node preparation.
void prepareFullReference(const CsrMatrix& a, const CsrMatrix& b,
                          const std::filesystem::path& directory, ValidationDiagnostics& diagnostics);
