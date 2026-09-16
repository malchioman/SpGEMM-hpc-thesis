#pragma once

#include "process_grid.hpp"

#include <unordered_map>

namespace trident {

struct MatrixBlock {
    RowBlock rows;
    RowBlock cols;
};

// Coarse 2D tile, then a contiguous row slice within that tile.
MatrixBlock matrixBlock(int rows, int cols, const ProcessGrid& grid, int i, int j, int k);
CsrMatrix sliceBlock(const CsrMatrix& matrix, MatrixBlock block);
CsrMatrix distributeBlocks(const CsrMatrix* matrix, int rows, int cols, const ProcessGrid& grid);
CsrMatrix gatherBlocks(const CsrMatrix& local, int rows, int cols, const ProcessGrid& grid);
int mpiCount(std::size_t size);

class LocalAccumulator {
public:
    LocalAccumulator(int rows, int cols);
    void addProduct(const CsrMatrix& a, const CsrMatrix& b);
    CsrMatrix finish() const;

private:
    int cols_;
    std::vector<std::unordered_map<int, double>> rows_;
};

}  // namespace trident
