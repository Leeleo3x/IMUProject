cmake_minimum_required(VERSION 3.0)

project (SpeedRegression)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(GLOG REQUIRED)
find_package(GFLAGS REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_library(SpeedRegression speed_regression.h speed_regression.cpp)
target_link_libraries(SpeedRegression ${OpenCV_LIBS} ${GLOG_LIBRARY})

add_executable(SpeedRegression_cli main.cpp)
target_link_libraries(SpeedRegression_cli SpeedRegression ${GFLAGS_LIBRARY} IMUUtility)
