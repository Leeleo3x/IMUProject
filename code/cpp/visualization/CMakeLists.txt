cmake_minimum_required(VERSION 3.0)

project(visualization)

find_package(Eigen REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMesh REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_library(IMUVisualization imu_visualization.h imu_visualization.cpp)
target_link_libraries(IMUVisualization ${OPENMESH_LIBRARIES} ${GLOG_LIBRARIES} ${OpenGL_LIBRARIES} ${GLUT_LIBRARIES})
add_executable(VisualizeTrajectory visualize_trajectory.cpp)

target_link_libraries(VisualizeTrajectory ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY})