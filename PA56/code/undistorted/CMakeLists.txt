cmake_minimum_required(VERSION 3.13)

project(undistort_image)

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

ADD_EXECUTABLE(undistort_image undistort_image.cpp)
TARGET_LINK_LIBRARIES(undistort_image ${OpenCV_LIBS})