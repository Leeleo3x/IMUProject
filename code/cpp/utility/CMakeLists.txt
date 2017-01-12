cmake_minimum_required(VERSION 3.0)

project(IMUUtility)

find_package(Eigen REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})
add_library(QuaternionUtility quaternion_spline.cpp quaternion_spline.h)

target_link_libraries(QuaternionUtility glog)