# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/liuhongwei/下载/clion-2018.3.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/liuhongwei/下载/clion-2018.3.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liuhongwei/workspace/slam/PA10/L7/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuhongwei/workspace/slam/PA10/L7/code/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/dBA.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dBA.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dBA.dir/flags.make

CMakeFiles/dBA.dir/directBA.cpp.o: CMakeFiles/dBA.dir/flags.make
CMakeFiles/dBA.dir/directBA.cpp.o: ../directBA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liuhongwei/workspace/slam/PA10/L7/code/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dBA.dir/directBA.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dBA.dir/directBA.cpp.o -c /home/liuhongwei/workspace/slam/PA10/L7/code/directBA.cpp

CMakeFiles/dBA.dir/directBA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dBA.dir/directBA.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liuhongwei/workspace/slam/PA10/L7/code/directBA.cpp > CMakeFiles/dBA.dir/directBA.cpp.i

CMakeFiles/dBA.dir/directBA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dBA.dir/directBA.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liuhongwei/workspace/slam/PA10/L7/code/directBA.cpp -o CMakeFiles/dBA.dir/directBA.cpp.s

# Object files for target dBA
dBA_OBJECTS = \
"CMakeFiles/dBA.dir/directBA.cpp.o"

# External object files for target dBA
dBA_EXTERNAL_OBJECTS =

dBA: CMakeFiles/dBA.dir/directBA.cpp.o
dBA: CMakeFiles/dBA.dir/build.make
dBA: /home/liuhongwei/workspace/slam/slambook/3rdparty/Sophus/build/libSophus.so
dBA: /usr/local/lib/libopencv_stitching.so.4.0.0
dBA: /usr/local/lib/libopencv_video.so.4.0.0
dBA: /usr/local/lib/libopencv_objdetect.so.4.0.0
dBA: /usr/local/lib/libopencv_photo.so.4.0.0
dBA: /usr/local/lib/libopencv_gapi.so.4.0.0
dBA: /usr/local/lib/libopencv_calib3d.so.4.0.0
dBA: /usr/local/lib/libopencv_ml.so.4.0.0
dBA: /usr/local/lib/libopencv_dnn.so.4.0.0
dBA: /home/liuhongwei/ORB_SLAM2/Pangolin/build/src/libpangolin.so
dBA: /usr/local/lib/libopencv_features2d.so.4.0.0
dBA: /usr/local/lib/libopencv_flann.so.4.0.0
dBA: /usr/local/lib/libopencv_highgui.so.4.0.0
dBA: /usr/local/lib/libopencv_videoio.so.4.0.0
dBA: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
dBA: /usr/local/lib/libopencv_imgproc.so.4.0.0
dBA: /usr/local/lib/libopencv_core.so.4.0.0
dBA: /usr/lib/x86_64-linux-gnu/libGLU.so
dBA: /usr/lib/x86_64-linux-gnu/libGL.so
dBA: /usr/lib/x86_64-linux-gnu/libGLEW.so
dBA: /usr/lib/x86_64-linux-gnu/libSM.so
dBA: /usr/lib/x86_64-linux-gnu/libICE.so
dBA: /usr/lib/x86_64-linux-gnu/libX11.so
dBA: /usr/lib/x86_64-linux-gnu/libXext.so
dBA: /usr/lib/x86_64-linux-gnu/libdc1394.so
dBA: /usr/lib/x86_64-linux-gnu/libavcodec.so
dBA: /usr/lib/x86_64-linux-gnu/libavformat.so
dBA: /usr/lib/x86_64-linux-gnu/libavutil.so
dBA: /usr/lib/x86_64-linux-gnu/libswscale.so
dBA: /usr/lib/x86_64-linux-gnu/libavdevice.so
dBA: /usr/lib/x86_64-linux-gnu/libpng.so
dBA: /usr/lib/x86_64-linux-gnu/libz.so
dBA: /usr/lib/x86_64-linux-gnu/libjpeg.so
dBA: /usr/lib/x86_64-linux-gnu/libtiff.so
dBA: /usr/lib/x86_64-linux-gnu/libIlmImf.so
dBA: CMakeFiles/dBA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liuhongwei/workspace/slam/PA10/L7/code/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dBA"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dBA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dBA.dir/build: dBA

.PHONY : CMakeFiles/dBA.dir/build

CMakeFiles/dBA.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dBA.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dBA.dir/clean

CMakeFiles/dBA.dir/depend:
	cd /home/liuhongwei/workspace/slam/PA10/L7/code/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuhongwei/workspace/slam/PA10/L7/code /home/liuhongwei/workspace/slam/PA10/L7/code /home/liuhongwei/workspace/slam/PA10/L7/code/cmake-build-debug /home/liuhongwei/workspace/slam/PA10/L7/code/cmake-build-debug /home/liuhongwei/workspace/slam/PA10/L7/code/cmake-build-debug/CMakeFiles/dBA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dBA.dir/depend
