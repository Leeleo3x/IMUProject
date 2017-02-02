cmake_minimum_required(VERSION 3.0)

project(visualization)

find_package(Pangolin REQUIRED)
find_package(Eigen REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(VisualizeTrajectory visualize_trajectory.cpp)

target_link_libraries(VisualizeTrajectory ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY})