#include "gemm.h"

#include <cstddef>
#include <mkl.h>
#include <vector>

void gemmVanilla(const std::vector<float> &matA, const std::vector<float> &matB,
                 std::vector<float> &matC, size_t matSize) {
  // Clear matC.
  for (size_t i = 0; i < matC.size(); i++) {
    matC[i] = 0;
  }

  // Matrix matrix multiply.
  for (size_t i = 0; i < matSize; i++) {
    for (size_t j = 0; j < matSize; j++) {
      size_t cIdx = i * matSize + j;
      for (size_t k = 0; k < matSize; k++) {
        size_t aIdx = i * matSize + k;
        size_t bIdx = k * matSize + j;
        matC[cIdx] += matA[aIdx] * matB[bIdx];
      }
    }
  }
}

void gemmVanillaParallel(const std::vector<float> &matA,
                         const std::vector<float> &matB,
                         std::vector<float> &matC, size_t matSize) {
  // Clear matC.
#pragma omp parallel for
  for (size_t i = 0; i < matC.size(); i++) {
    matC[i] = 0;
  }

  // Matrix matrix multiply.
#pragma omp parallel for
  for (size_t i = 0; i < matSize; i++) {
    for (size_t j = 0; j < matSize; j++) {
      size_t cIdx = i * matSize + j;
      for (size_t k = 0; k < matSize; k++) {
        size_t aIdx = i * matSize + k;
        size_t bIdx = k * matSize + j;
        matC[cIdx] += matA[aIdx] * matB[bIdx];
      }
    }
  }
}

void gemmTranspose(const std::vector<float> &matA,
                   const std::vector<float> &matB, std::vector<float> &matC,
                   size_t matSize) {
  // Transpose matrix B.
  std::vector<float> matBTrans(matB.size());
  mkl_somatcopy('r', 't', matSize, matSize, 1, matB.data(), matSize,
                matBTrans.data(), matSize);

  // Clear matC.
#pragma omp parallel for
  for (size_t i = 0; i < matC.size(); i++) {
    matC[i] = 0;
  }

  // Matrix matrix multiply.
#pragma omp parallel for
  for (size_t i = 0; i < matSize; i++) {
    for (size_t j = 0; j < matSize; j++) {
      size_t cIdx = i * matSize + j;
      for (size_t k = 0; k < matSize; k++) {
        size_t aIdx = i * matSize + k;
        size_t bIdx = j * matSize + k;
        matC[cIdx] += matA[aIdx] * matBTrans[bIdx];
      }
    }
  }
}

void gemmBlock(const std::vector<float> &matA, const std::vector<float> &matB,
               std::vector<float> &matC, size_t matSize, size_t blockSize) {
  size_t blockNum = matSize / blockSize;

  // Clear matC.
#pragma omp parallel for
  for (size_t i = 0; i < matC.size(); i++) {
    matC[i] = 0;
  }

  // Traverse blocks.
#pragma omp parallel for
  for (size_t bi = 0; bi < blockNum; bi++) {
    for (size_t bj = 0; bj < blockNum; bj++) {
      for (size_t bk = 0; bk < blockNum; bk++) {
        // Block GEMM.
        for (size_t i = 0; i < blockSize; i++) {
          for (size_t j = 0; j < blockSize; j++) {
            for (size_t k = 0; k < blockSize; k++) {
              size_t aIdx = bi * blockSize * blockNum * blockSize +
                            i * blockNum * blockSize + bk * blockSize + k;
              size_t bIdx = bk * blockSize * blockNum * blockSize +
                            k * blockNum * blockSize + bj * blockSize + j;
              size_t cIdx = bi * blockSize * blockNum * blockSize +
                            i * blockNum * blockSize + bj * blockSize + j;
              matC[cIdx] += matA[aIdx] * matB[bIdx];
            }
          }
        }
      }
    }
  }
}
