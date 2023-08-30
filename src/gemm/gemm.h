#pragma once

#include <cstddef>
#include <vector>

void gemmVanilla(const std::vector<float> &matA, const std::vector<float> &matB,
                 std::vector<float> &matC, size_t matSize);

void gemmVanillaParallel(const std::vector<float> &matA,
                         const std::vector<float> &matB,
                         std::vector<float> &matC, size_t matSize);

void gemmTranspose(const std::vector<float> &matA,
                   const std::vector<float> &matB, std::vector<float> &matC,
                   size_t matSize);

void gemmBlock(const std::vector<float> &matA, const std::vector<float> &matB,
               std::vector<float> &matC, size_t matSize, size_t blockSize);

void gemmBlockTranspose(const std::vector<float> &matA,
                        const std::vector<float> &matB,
                        std::vector<float> &matC, size_t matSize,
                        size_t blockSize);

void gemmBlockTransposeCopy(const std::vector<float> &matA,
                            const std::vector<float> &matB,
                            std::vector<float> &matC, size_t matSize);

void gemmBlockCopy(const std::vector<float> &matA,
                   const std::vector<float> &matB, std::vector<float> &matC,
                   size_t matSize);
