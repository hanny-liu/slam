project(pointcloud)

find_package(PCL REQUIRED COMPONENTS common io)

find_package(OpenCV REQUIRED)

ADD_DEFINITIONS(${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})

ADD_EXECUTABLE(generate_pointcloud generatePointCloud.cpp)