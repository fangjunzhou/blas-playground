find_package(benchmark CONFIG REQUIRED)

file(
  GLOB_RECURSE
  BENCHMARK_SRC
  *.h
  *.cpp
)
add_executable(
  benchmark
  ${BENCHMARK_SRC}
)
target_include_directories(
  benchmark
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(
  benchmark
  PRIVATE
  gemm
  benchmark::benchmark
  MKL::MKL
)
