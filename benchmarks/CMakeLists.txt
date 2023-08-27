find_package(benchmark CONFIG REQUIRED)
find_package(MKL CONFIG REQUIRED)

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
target_link_libraries(
  benchmark
  PRIVATE
  benchmark::benchmark
  MKL::MKL
)
