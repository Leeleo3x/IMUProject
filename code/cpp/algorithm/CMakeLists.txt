cmake_minimum_required(VERSION 3.0)

project(IMUAlgorithm)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)

add_library(IMUAlgorithm double_intergration.cpp double_intergration.h)

target_link_libraries(IMUAlgorithm ${OpenCV_LIBS} ${GLOG_LIBRARIES})