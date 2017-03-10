cmake_minimum_required(VERSION 3.0)

project(IMUUtility)

find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_library(IMUUtility data_io.cpp data_io.h utility.h utility.cpp)
target_link_libraries(IMUUtility ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${OPENMESH_LIBRARIES})

#add_executable(IMUUtility_cli main.cpp)
#target_link_libraries(IMUUtility_cli IMUUtility)
