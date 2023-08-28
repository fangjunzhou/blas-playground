#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdlib>
#include <mkl.h>
#include <mkl_cblas.h>
#include <vector>

#include "gemm/gemm.h"

std::vector<float> matA;
std::vector<float> matB;
std::vector<float> matC;

static void GemmSetup(const benchmark::State &state) {
  size_t matSize = state.range(0);
  // Init random input matricies.
  matA.resize(matSize * matSize);
  matB.resize(matSize * matSize);
  for (size_t i = 0; i < matSize; i++) {
    for (size_t j = 0; j < matSize; j++) {
      size_t idx = i * matSize + j;
      matA[idx] = (float)std::rand() / (float)RAND_MAX;
      matB[idx] = (float)std::rand() / (float)RAND_MAX;
    }
  }
  matC.resize(matSize * matSize);
}

static void mkl_gemm_benchmark(benchmark::State &state) {
  size_t matSize = state.range(0);
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matSize, matSize,
                matSize, 1, matA.data(), matSize, matB.data(), matSize, 0,
                matC.data(), matSize);
  }
}

BENCHMARK(mkl_gemm_benchmark)
    ->Setup(GemmSetup)
    ->RangeMultiplier(2)
    ->Range(2, 1024);

static void gemmVanilla_benchmark(benchmark::State &state) {
  size_t matSize = state.range(0);
  for (auto _ : state) {
    gemmVanilla(matA, matB, matC, matSize);
  }
}

BENCHMARK(gemmVanilla_benchmark)
    ->Setup(GemmSetup)
    ->RangeMultiplier(2)
    ->Range(2, 1024);

static void gemmVanillaParallel_benchmark(benchmark::State &state) {
  size_t matSize = state.range(0);
  for (auto _ : state) {
    gemmVanillaParallel(matA, matB, matC, matSize);
  }
}

BENCHMARK(gemmVanillaParallel_benchmark)
    ->Setup(GemmSetup)
    ->RangeMultiplier(2)
    ->Range(2, 1024);
