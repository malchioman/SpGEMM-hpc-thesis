#pragma once

#include "csr_matrix.hpp"

#include <string>

// Reads a Matrix Market coordinate matrix and converts it to CSR.
CsrMatrix readMatrixMarket(const std::string& path);
