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
            size_t cIdx = bi * blockSize * blockNum * blockSize +
                          i * blockNum * blockSize + bj * blockSize + j;
            float partial = 0;
#pragma omp simd reduction(+ : partial)
            for (size_t k = 0; k < blockSize; k++) {
              size_t aIdx = bi * blockSize * blockNum * blockSize +
                            i * blockNum * blockSize + bk * blockSize + k;
              size_t bIdx = bk * blockSize * blockNum * blockSize +
                            k * blockNum * blockSize + bj * blockSize + j;
              partial += matA[aIdx] * matB[bIdx];
            }
            matC[cIdx] += partial;
          }
        }
      }
    }
  }
}

void gemmBlockTranspose(const std::vector<float> &matA,
                        const std::vector<float> &matB,
                        std::vector<float> &matC, size_t matSize,
                        size_t blockSize) {
  // Transpose matrix B.
  std::vector<float> matBTrans(matB.size());
  mkl_somatcopy('r', 't', matSize, matSize, 1, matB.data(), matSize,
                matBTrans.data(), matSize);

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
            size_t cIdx = bi * blockSize * blockNum * blockSize +
                          i * blockNum * blockSize + bj * blockSize + j;
            float partial = 0;
#pragma omp simd reduction(+ : partial)
            for (size_t k = 0; k < blockSize; k++) {
              size_t aIdx = bi * blockSize * blockNum * blockSize +
                            i * blockNum * blockSize + bk * blockSize + k;
              size_t bIdx = bj * blockSize * blockNum * blockSize +
                            j * blockNum * blockSize + bk * blockSize + k;
              partial += matA[aIdx] * matBTrans[bIdx];
            }
            matC[cIdx] += partial;
          }
        }
      }
    }
  }
}

#define BLOCK_SIZE 16
alignas(64) float localA[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localB[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localC[BLOCK_SIZE][BLOCK_SIZE];

#pragma omp threadprivate(localA, localB, localC)

void gemmBlockTransposeCopy(const std::vector<float> &matA,
                            const std::vector<float> &matB,
                            std::vector<float> &matC, size_t matSize) {
  // Transpose matrix B.
  std::vector<float> matBTrans(matB.size());
  mkl_somatcopy('r', 't', matSize, matSize, 1, matB.data(), matSize,
                matBTrans.data(), matSize);

  size_t blockNum = matSize / BLOCK_SIZE;

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
        // Copy local block.
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t j = 0; j < BLOCK_SIZE; j++) {
            size_t aIdx = bi * BLOCK_SIZE * blockNum * BLOCK_SIZE +
                          i * blockNum * BLOCK_SIZE + bk * BLOCK_SIZE + j;
            size_t bIdx = bj * BLOCK_SIZE * blockNum * BLOCK_SIZE +
                          i * blockNum * BLOCK_SIZE + bk * BLOCK_SIZE + j;
            localA[i][j] = matA[aIdx];
            localB[i][j] = matBTrans[bIdx];
            localC[i][j] = 0;
          }
        }

        // Block GEMM.
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t j = 0; j < BLOCK_SIZE; j++) {
#pragma omp simd
            for (size_t k = 0; k < BLOCK_SIZE; k++) {
              localC[i][j] += localA[i][k] * localB[j][k];
            }
          }
        }

        // Copy localC back.
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t j = 0; j < BLOCK_SIZE; j++) {
            size_t cIdx = bi * BLOCK_SIZE * blockNum * BLOCK_SIZE +
                          i * blockNum * BLOCK_SIZE + bj * BLOCK_SIZE + j;
            matC[cIdx] += localC[i][j];
          }
        }
      }
    }
  }
}
