#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <mkl.h>
#include <mkl_cblas.h>
#include <vector>

#include "gemm/gemm.h"

#define MAT_SIZE 32
#define BLOCK_SIZE 8
#define ALLOWED_ERR 1e-5

class BlasLevel3Test : public ::testing::Test {
protected:
  std::vector<float> matA;
  std::vector<float> matB;
  std::vector<float> matC;
  std::vector<float> matCRef;

  void SetUp() override {
    this->matA.resize(MAT_SIZE * MAT_SIZE);
    this->matB.resize(MAT_SIZE * MAT_SIZE);
    for (size_t i = 0; i < MAT_SIZE; i++) {
      for (size_t j = 0; j < MAT_SIZE; j++) {
        size_t idx = i * MAT_SIZE + j;
        matA[idx] = (float)std::rand() / (float)RAND_MAX;
        matB[idx] = (float)std::rand() / (float)RAND_MAX;
      }
    }
    this->matC.resize(MAT_SIZE * MAT_SIZE);
    this->matCRef.resize(MAT_SIZE * MAT_SIZE);
  }
};

TEST_F(BlasLevel3Test, gemmVanilla) {
  // MKL reference
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE,
              MAT_SIZE, 1, this->matA.data(), MAT_SIZE, this->matB.data(),
              MAT_SIZE, 0, this->matCRef.data(), MAT_SIZE);
  // Vanilla GEMM implementation.
  gemmVanilla(this->matA, this->matB, this->matC, MAT_SIZE);

  // Compare the result.
  for (size_t i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
    EXPECT_NEAR(matC[i], matCRef[i], ALLOWED_ERR);
  }
}

TEST_F(BlasLevel3Test, gemmVanillaParallel) {
  // MKL reference
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE,
              MAT_SIZE, 1, this->matA.data(), MAT_SIZE, this->matB.data(),
              MAT_SIZE, 0, this->matCRef.data(), MAT_SIZE);
  // Vanilla GEMM implementation.
  gemmVanillaParallel(this->matA, this->matB, this->matC, MAT_SIZE);

  // Compare the result.
  for (size_t i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
    EXPECT_NEAR(matC[i], matCRef[i], ALLOWED_ERR);
  }
}

TEST_F(BlasLevel3Test, gemmTranspose) {
  // MKL reference
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE,
              MAT_SIZE, 1, this->matA.data(), MAT_SIZE, this->matB.data(),
              MAT_SIZE, 0, this->matCRef.data(), MAT_SIZE);
  // Transpose GEMM implementation.
  gemmTranspose(this->matA, this->matB, this->matC, MAT_SIZE);

  // Compare the result.
  for (size_t i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
    EXPECT_NEAR(matC[i], matCRef[i], ALLOWED_ERR);
  }
}

TEST_F(BlasLevel3Test, gemmBlock) {
  // MKL reference
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE,
              MAT_SIZE, 1, this->matA.data(), MAT_SIZE, this->matB.data(),
              MAT_SIZE, 0, this->matCRef.data(), MAT_SIZE);
  // Block GEMM implementation.
  gemmBlock(this->matA, this->matB, this->matC, MAT_SIZE, BLOCK_SIZE);

  // Compare the result.
  for (size_t i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
    EXPECT_NEAR(matC[i], matCRef[i], ALLOWED_ERR);
  }
}
