cmake_minimum_required(VERSION 3.0)

project(IMUAlgorithm)

find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})
add_library(IMUAlgorithm temporal_function.h
        position_function.h position_function.cpp
        rotation_function.h rotation_function.cpp)

target_link_libraries(IMUAlgorithm glog)