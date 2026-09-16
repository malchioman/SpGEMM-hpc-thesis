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

struct ProductResult {
    CsrMatrix matrix;
    double interNodeSeconds = 0.0;
    double intraNodeSeconds = 0.0;
    double computeSeconds = 0.0;
    double totalSeconds = 0.0;
};

// Static owners and the staggered schedule are shared by every intra-node backend.
ProductResult multiply(const CsrMatrix& a, const CsrMatrix& b, const ProcessGrid& grid,
                       const ExecutionPlan& plan, IntraNodeExchange& exchange,
                       ProductWorkspace& workspace);

}  // namespace trident
