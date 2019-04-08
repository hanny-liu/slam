cmake_minimum_required(VERSION 3.13.2)
PROJECT(g2o)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#OPENCV
find_package(OpenCV 4.0.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

#G2O
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

list( APPEND CMAKE_MODULE_PATH /home/liuhongwei/g2o/cmake_modules )
set(G2O_ROOT /usr/local/include/g2o)

find_package( G2O REQUIRED )
include_directories(
        ${G2O_INCLUDE_DIRS}
)

#sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

#EIGEN
include_directories("/usr/include/eigen3")

FIND_PACKAGE( CSparse )
INCLUDE_DIRECTORIES( ${CSPARSE_INCLUDE_DIR} )


ADD_EXECUTABLE(BA BA.cpp)
TARGET_LINK_LIBRARIES(BA ${Sophus_LIBRARIES} ${OpenCV_LIBS} g2o_core g2o_stuff )
