#pragma once

#include "matrix_blocks.hpp"

#include <array>

namespace trident {

struct TileShape {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
};

struct Stage {
    int inner = 0;
    int aSource = 0;
    int aTarget = 0;
    int bSource = 0;
    int bTarget = 0;
    TileShape a;
    TileShape b;
    TileShape panel;
    std::vector<int> rowCounts;
    std::vector<int> rowOffsets;
    std::vector<int> nnzCounts;
    std::vector<int> nnzOffsets;
};

struct ExecutionPlan {
    std::vector<Stage> stages;
    int resultRows = 0;
    int resultCols = 0;
};

ExecutionPlan preparePlan(const CsrMatrix& a, const CsrMatrix& b, const ProcessGrid& grid);

class IntraNodeExchange {
public:
    virtual ~IntraNodeExchange() = default;
    virtual void assemble(const CsrMatrix& slice, const Stage& stage, CsrMatrix& panel) = 0;
};

struct ProductWorkspace {
    explicit ProductWorkspace(const ExecutionPlan& plan);
    CsrMatrix a;
    CsrMatrix b;
    CsrMatrix panel;
};

class InterNodeExchange {
public:
    virtual ~InterNodeExchange() = default;
    virtual void begin(const CsrMatrix& a, const CsrMatrix& b) = 0;
    virtual void fetch(const CsrMatrix& a, const CsrMatrix& b, const Stage& stage,
                       ProductWorkspace& workspace) = 0;
    virtual void finish() = 0;
    // Collective resource cleanup is explicit, never performed while unwinding exceptions.
    virtual void close() = 0;
};

struct ProductResult {
    CsrMatrix matrix;
    double interNodeSeconds = 0.0;
    double intraNodeSeconds = 0.0;
    double computeSeconds = 0.0;
    double totalSeconds = 0.0;
};

// Static owners and the staggered schedule are shared by all communication backends.
ProductResult multiply(const CsrMatrix& a, const CsrMatrix& b, const ExecutionPlan& plan,
                       IntraNodeExchange& exchange, ProductWorkspace& workspace,
                       InterNodeExchange& inter);

}  // namespace trident
