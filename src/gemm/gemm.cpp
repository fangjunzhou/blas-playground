#include "gemm.h"
#include <cstddef>

void gemm_vanilla(const std::vector<float> &matA,
                  const std::vector<float> &matB, std::vector<float> &matC,
                  size_t matSize) {
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
