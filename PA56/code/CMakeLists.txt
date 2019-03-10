cmake_minimum_required( VERSION 2.8 )
project( gaussnewton )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

# eigen
include_directories( "/usr/include/eigen3/" )

add_executable( gaussnewton gaussnewton.cpp )

# 与OpenCV链接
target_link_libraries( gaussnewton  ${OpenCV_LIBS} )