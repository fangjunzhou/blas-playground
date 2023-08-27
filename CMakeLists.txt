cmake_minimum_required(VERSION 3.27.1)

# Setup vcpkg
set(
  CMAKE_TOOLCHAIN_FILE
  ${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file"
)

project(gemm-playground)

# --------------------- Project Settings --------------------- #

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Project binary and library output directory.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "Library Output" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "Binary Output" FORCE)

# ---------------------- Build Project  ---------------------- #

# Build src
add_subdirectory(src)

# Build benchmark
add_subdirectory(benchmarks)
