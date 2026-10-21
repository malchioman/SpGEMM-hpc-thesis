#pragma once

#include "csr_matrix.hpp"

#include <filesystem>
#include <string>
#include <vector>

// Only the serial reference, never a previous validation outcome.
struct PartialReference {
    int rows = 0;
    int cols = 0;
    bool finite = true;
    std::vector<CsrOffset> rowNnz;
};

struct ValidationDiagnostics {
    std::string source;
    std::string key;
    double identitySeconds = 0;
    double buildSeconds = 0;
    double readSeconds = 0;
    double compareSeconds = 0;
    double writeSeconds = 0;
};

// Same ordered numeric traversal as serialSpgemm and the original Partial check.
PartialReference buildPartialReference(const CsrMatrix& a, const CsrMatrix& b);
std::string partialReferenceKey(const CsrMatrix& a, const CsrMatrix& b);
std::string partialReferenceBuildId();
std::filesystem::path partialReferencePath(const std::filesystem::path& directory,
                                          const std::string& key);
PartialReference readPartialReference(const std::filesystem::path& directory,
                                     const std::string& key, int rows, int cols);
PartialReference preparePartialReference(const CsrMatrix& a, const CsrMatrix& b,
                                        const std::filesystem::path& directory,
                                        ValidationDiagnostics& diagnostics);
void printValidationDiagnostics(const ValidationDiagnostics& diagnostics);

// Binary codec exposed for format/64-bit regression tests, not reference injection.
std::string encodePartialReference(const PartialReference& reference, const std::string& key);
PartialReference decodePartialReference(const std::string& bytes, const std::string& key,
                                        int rows, int cols);
