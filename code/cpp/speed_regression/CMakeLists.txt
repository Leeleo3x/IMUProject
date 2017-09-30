cmake_minimum_required(VERSION 3.0)

project (SpeedRegression)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_library(SpeedRegression speed_regression.h speed_regression.cpp
        model_wrapper.h model_wrapper.cpp)
target_link_libraries(SpeedRegression ${OpenCV_LIBS} ${GLOG_LIBRARY})

add_executable(SpeedRegression_cli main.cpp)
target_link_libraries(SpeedRegression_cli SpeedRegression ${GFLAGS_LIBRARY} IMUUtility)
