enable_testing()

find_package(GTest CONFIG REQUIRED)

file(
  GLOB_RECURSE
  TEST_SRC
  *.h
  *.cpp
)
add_executable(
  test
  ${TEST_SRC}
)
target_include_directories(
  test
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(
  test
  PRIVATE
  gemm
  MKL::MKL
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  GTest::gmock_main
)

add_test(AllTestsInMain test)
