FIND_PATH(HELLO_INCLUDE_DIR hello.h /home/liuhongwei/workspace/slam/hellolib/src)
FIND_LIBRARY(HELLO_LIBRARY NAMES HELLO PATH /home/liuhongwei/workspace/slam/hello/src)
IF(HELLO_INCLUDE_DIR AND HELLO_LIBRARY)
 SET(HELLO_FOUND TRUE)
ENDIF(HELLO_INCLUDE_DIR AND HELLO_LIBRARY)
IF(HELLO_FOUND)
 IF(NOT HELLO_FIND_QUIETLY)
   MESSAGE(STATUS"FOUND HELLO:${HELLO_LIBRARY}")
 ENDIF(NOT HELLO_FIND_QUIETLY)
ELSE(HELLO_FOUND)
 IF(HELLO_FIND_REQUIRED)
   MESSAGE(FATA_ERROR "COULD NOT FIND HELLO LIBRARY")
 ENDIF(HELLO_FIND_REQUIRED)
ENDIF(HELLO_FOUND)
