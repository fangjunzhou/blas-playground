cmake_minimum_required(VERSION 3.27.1)

# Setup vcpkg
set(
  CMAKE_TOOLCHAIN_FILE
  ${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file"
)

project(blas-playground)

# --------------------- Project Settings --------------------- #

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Project binary and library output directory.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "Library Output" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "Binary Output" FORCE)
# Optimization level.
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ----------------------- Dependencies ----------------------- #

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Intel MKL
find_package(MKL CONFIG REQUIRED)

# ---------------------- Build Project  ---------------------- #

# Build src
add_subdirectory(src)

# Build tests
add_subdirectory(tests)

# Build benchmarks
add_subdirectory(benchmarks)
