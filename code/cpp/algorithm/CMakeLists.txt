cmake_minimum_required(VERSION 3.0)

project(IMUAlgorithm)

find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)

include_directories(${EIGEN_INCLUDE_DIR})
add_library(algorithm geometry.cc geometry.h)

target_link_libraries(algorithm ${GLOG_LIBRARIES})

# test
enable_testing()
add_executable(geometry_test_cli ${PROJECT_ROOT}/googletest/src/gtest_main.cc geometry_test.cc)
target_link_libraries(geometry_test_cli algorithm gtest)
add_test(NAME geometry_test COMMAND geometry_test_cli)