ADD_LIBRARY(HELLO SHARED function.cpp)
SET_TARGET_PROPERTIES(HELLO PROPERTIES CLEAN_DIRECT_OUTPUT 1)
ADD_LIBRARY(HELLO_STATIC STATIC function.cpp)
SET_TARGET_PROPERTIES(HELLO_STATIC PROPERTIES OUTPUT_NAME "HELLO")
SET_TARGET_PROPERTIES(HELLO_STATIC PROPERTIES CLEAN_DIRECT_OUTPUT 1)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/include)
INSTALL(TARGETS HELLO HELLO_STATIC
               LIBRARY DESTINATION /home/liuhongwei/workspace/slam/hello/include
               ARCHIVE DESTINATION /home/liuhongwei/workspace/slam/hello/include)

INSTALL(DIRECTORY cmake/ DESTINATION /home/liuhongwei/workspace/slam/hello/cmake)
SET(CMAKE_MODULE_PATH /home/liuhongwei/workspace/slam/hello/cmake)
