#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdlib>
#include <mkl.h>
#include <mkl_cblas.h>
#include <vector>

static void mkl_gemm_benchmark(benchmark::State &state) {
  size_t matSize = state.range(0);
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
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matSize, matSize,
                matSize, 1, matA.data(), matSize, matB.data(), matSize, 0,
                matC.data(), matSize);
  }
}

BENCHMARK(mkl_gemm_benchmark)->RangeMultiplier(2)->Range(2, 1024);
