cmake_minimum_required(VERSION 3.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(gflags REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(algorithm)
add_subdirectory(utility)

add_subdirectory(genDataSet)

