cmake_minimum_required(VERSION 3.0)

project(visualization)


find_package(Eigen REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_library(IMUVisualization imu_visualization.h imu_visualization.cpp)
target_link_libraries(IMUVisualization ${OPENMESH_LIBRARIES} ${GLOG_LIBRARIES})