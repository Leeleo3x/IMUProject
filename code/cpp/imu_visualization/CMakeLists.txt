cmake_minimum_required(VERSION 3.0)

set(CMAKE_PREFIX_PATH /opt/Qt5/5.8/gcc_64)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR  ON)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(Eigen REQUIRED)

include_directories(${EIGEN_INCLUDE_DIR})

add_executable(IMU_visualization imuvis_cli.cpp main_widget.cpp main_widget.h
        renderable.cpp renderable.h)

target_link_libraries(IMU_visualization IMUUtility Qt5::Core Qt5::Widgets Qt5::Gui
        ${GLOG_LIBRARY} ${GFLAGS_LIBRARY})