cmake_minimum_required(VERSION 3.0)

project(IMUUtility)

find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_library(IMUUtility imu_dataset.cpp imu_dataset.h)

target_link_libraries(IMUUtility ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES})
