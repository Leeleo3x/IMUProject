cmake_minimum_required(VERSION 3.0)

project(visualization)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_PREFIX_PATH /opt/Qt/5.8/gcc_64)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Eigen REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMesh REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})
find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui OpenGL)

add_library(IMUVisualization imu_visualization.h imu_visualization.cpp renderable.h renderable.cpp)
target_link_libraries(IMUVisualization ${OPENMESH_LIBRARIES} ${GLOG_LIBRARIES}
        Qt5::Widgets Qt5::OpenGL Qt5::Core Qt5::Gui)

add_executable(VisualizeTrajectory visualize_trajectory.cpp)
target_link_libraries(VisualizeTrajectory ${OpenCV_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY})