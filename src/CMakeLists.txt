# GEMM.
file(
  GLOB_RECURSE
  GEMM_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.cpp
)
add_library(
  gemm
  ${GEMM_SRC}
)
