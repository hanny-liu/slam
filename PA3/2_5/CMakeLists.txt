cmake_minimum_required(VERSION 2.8)
project(2_5)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/include/eigen3)

add_executable(2_5 main.cpp)