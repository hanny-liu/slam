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
CMAKE_SOURCE_DIR = "/home/liuhongwei/workspace/slam/VIO program"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/VIO_program.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/VIO_program.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VIO_program.dir/flags.make

CMakeFiles/VIO_program.dir/main.cpp.o: CMakeFiles/VIO_program.dir/flags.make
CMakeFiles/VIO_program.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/VIO_program.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VIO_program.dir/main.cpp.o -c "/home/liuhongwei/workspace/slam/VIO program/main.cpp"

CMakeFiles/VIO_program.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VIO_program.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/liuhongwei/workspace/slam/VIO program/main.cpp" > CMakeFiles/VIO_program.dir/main.cpp.i

CMakeFiles/VIO_program.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VIO_program.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/liuhongwei/workspace/slam/VIO program/main.cpp" -o CMakeFiles/VIO_program.dir/main.cpp.s

CMakeFiles/VIO_program.dir/g2o.cpp.o: CMakeFiles/VIO_program.dir/flags.make
CMakeFiles/VIO_program.dir/g2o.cpp.o: ../g2o.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/VIO_program.dir/g2o.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VIO_program.dir/g2o.cpp.o -c "/home/liuhongwei/workspace/slam/VIO program/g2o.cpp"

CMakeFiles/VIO_program.dir/g2o.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VIO_program.dir/g2o.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/liuhongwei/workspace/slam/VIO program/g2o.cpp" > CMakeFiles/VIO_program.dir/g2o.cpp.i

CMakeFiles/VIO_program.dir/g2o.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VIO_program.dir/g2o.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/liuhongwei/workspace/slam/VIO program/g2o.cpp" -o CMakeFiles/VIO_program.dir/g2o.cpp.s

# Object files for target VIO_program
VIO_program_OBJECTS = \
"CMakeFiles/VIO_program.dir/main.cpp.o" \
"CMakeFiles/VIO_program.dir/g2o.cpp.o"

# External object files for target VIO_program
VIO_program_EXTERNAL_OBJECTS =

VIO_program: CMakeFiles/VIO_program.dir/main.cpp.o
VIO_program: CMakeFiles/VIO_program.dir/g2o.cpp.o
VIO_program: CMakeFiles/VIO_program.dir/build.make
VIO_program: /home/liuhongwei/workspace/slam/slambook/3rdparty/Sophus/build/libSophus.so
VIO_program: /usr/local/lib/libopencv_stitching.so.4.0.0
VIO_program: /usr/local/lib/libopencv_video.so.4.0.0
VIO_program: /usr/local/lib/libopencv_objdetect.so.4.0.0
VIO_program: /usr/local/lib/libopencv_photo.so.4.0.0
VIO_program: /usr/local/lib/libopencv_gapi.so.4.0.0
VIO_program: /usr/local/lib/libopencv_calib3d.so.4.0.0
VIO_program: /usr/local/lib/libopencv_ml.so.4.0.0
VIO_program: /usr/local/lib/libopencv_dnn.so.4.0.0
VIO_program: /usr/local/lib/libopencv_features2d.so.4.0.0
VIO_program: /usr/local/lib/libopencv_flann.so.4.0.0
VIO_program: /usr/local/lib/libopencv_highgui.so.4.0.0
VIO_program: /usr/local/lib/libopencv_videoio.so.4.0.0
VIO_program: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
VIO_program: /usr/local/lib/libopencv_imgproc.so.4.0.0
VIO_program: /usr/local/lib/libopencv_core.so.4.0.0
VIO_program: CMakeFiles/VIO_program.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable VIO_program"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VIO_program.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VIO_program.dir/build: VIO_program

.PHONY : CMakeFiles/VIO_program.dir/build

CMakeFiles/VIO_program.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VIO_program.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VIO_program.dir/clean

CMakeFiles/VIO_program.dir/depend:
	cd "/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/liuhongwei/workspace/slam/VIO program" "/home/liuhongwei/workspace/slam/VIO program" "/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug" "/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug" "/home/liuhongwei/workspace/slam/VIO program/cmake-build-debug/CMakeFiles/VIO_program.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/VIO_program.dir/depend
