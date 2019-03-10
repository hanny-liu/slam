PROJECT(newhello1)
ADD_EXECUTABLE(sayhello usehello.cpp)
INCLUDE_DIRECTORIES(/usr/local/include)
TARGET_LINK_LIBRARIES(sayhello hello)

