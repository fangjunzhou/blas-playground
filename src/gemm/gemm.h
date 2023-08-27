#pragma once

#include <cstddef>
#include <vector>

void gemm_vanilla(const std::vector<float> &matA,
                  const std::vector<float> &matB, std::vector<float> &matC,
                  size_t matSize);
