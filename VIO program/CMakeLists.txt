cmake_minimum_required(VERSION 3.13)
project(VIO_program)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

include_directories("/usr/include/eigen3")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH /home/liuhongwei/g2o/cmake_modules)
find_package(G2O REQUIRED)
include_directories(${G20_INCLUDE_DIRS})



add_executable(VIO_program main.cpp g2o.cpp)
TARGET_LINK_LIBRARIES(VIO_program ${Sophus_LIBRARIES} ${OpenCV_LIBS} g2o_core g2o_stuff)