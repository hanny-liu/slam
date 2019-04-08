cmake_minimum_required(VERSION 3.13)

PROJECT(optical_flow)

SET(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES("/usr/include/eigen3/")

find_package(Sophus REQUIRED)

include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})


ADD_EXECUTABLE(optical_flow optical_flow.cpp)
TARGET_LINK_LIBRARIES(optical_flow ${OpenCV_LIBS})

ADD_EXECUTABLE(direct_method direct_method.cpp)
TARGET_LINK_LIBRARIES(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES})

ADD_EXECUTABLE(Disparity Disparity.cpp)
TARGET_LINK_LIBRARIES(Disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES})