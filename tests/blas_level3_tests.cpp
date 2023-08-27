#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <mkl.h>
#include <mkl_cblas.h>
#include <vector>

#include "gemm/gemm.h"

TEST(blas_leve3_test, gemm_vanilla) {
  // The size of the matrix.
  const size_t matSize = 16;
  // Allowed error size.
  const float allowedErr = 1e-5;

  // Init random input matricies.
  std::vector<float> matA(matSize * matSize);
  std::vector<float> matB(matSize * matSize);
  for (size_t i = 0; i < matSize; i++) {
    for (size_t j = 0; j < matSize; j++) {
      size_t idx = i * matSize + j;
      matA[idx] = (float)std::rand() / (float)RAND_MAX;
      matB[idx] = (float)std::rand() / (float)RAND_MAX;
    }
  }
  // Output matrix.
  std::vector<float> matC(matSize * matSize);
  // MKL reference matrix.
  std::vector<float> matCRef(matSize * matSize);

  // MKL reference
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matSize, matSize,
              matSize, 1, matA.data(), matSize, matB.data(), matSize, 0,
              matCRef.data(), matSize);
  // Vanilla GEMM implementation.
  gemm_vanilla(matA, matB, matC, matSize);

  // Compare the result.
  for (size_t i = 0; i < matSize * matSize; i++) {
    EXPECT_NEAR(matC[i], matCRef[i], allowedErr);
  }
}
