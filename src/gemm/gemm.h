#pragma once

#include <cstddef>
#include <vector>

void gemmVanilla(const std::vector<float> &matA, const std::vector<float> &matB,
                 std::vector<float> &matC, size_t matSize);

void gemmVanillaParallel(const std::vector<float> &matA,
                         const std::vector<float> &matB,
                         std::vector<float> &matC, size_t matSize);
