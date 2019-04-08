PROJECT(HELLOLIB)
ADD_SUBDIRECTORY(src bin)
#INSTALL(TARGETS HELLO HELLO
#           LIBRARY DESTINATION /home/liuhongwei/workspace/slam/hello/include
#           ARCHIVE DESTINATION /home/liuhongwei/workspace/slam/hello/include)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake})
