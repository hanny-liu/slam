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
CMAKE_SOURCE_DIR = /home/liuhongwei/workspace/slam/PA56/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuhongwei/workspace/slam/PA56/code

# Include any dependencies generated for this target.
include CMakeFiles/gaussnewton.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gaussnewton.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gaussnewton.dir/flags.make

CMakeFiles/gaussnewton.dir/gaussnewton.cpp.o: CMakeFiles/gaussnewton.dir/flags.make
CMakeFiles/gaussnewton.dir/gaussnewton.cpp.o: gaussnewton.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liuhongwei/workspace/slam/PA56/code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gaussnewton.dir/gaussnewton.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gaussnewton.dir/gaussnewton.cpp.o -c /home/liuhongwei/workspace/slam/PA56/code/gaussnewton.cpp

CMakeFiles/gaussnewton.dir/gaussnewton.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gaussnewton.dir/gaussnewton.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liuhongwei/workspace/slam/PA56/code/gaussnewton.cpp > CMakeFiles/gaussnewton.dir/gaussnewton.cpp.i

CMakeFiles/gaussnewton.dir/gaussnewton.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gaussnewton.dir/gaussnewton.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liuhongwei/workspace/slam/PA56/code/gaussnewton.cpp -o CMakeFiles/gaussnewton.dir/gaussnewton.cpp.s

# Object files for target gaussnewton
gaussnewton_OBJECTS = \
"CMakeFiles/gaussnewton.dir/gaussnewton.cpp.o"

# External object files for target gaussnewton
gaussnewton_EXTERNAL_OBJECTS =

gaussnewton: CMakeFiles/gaussnewton.dir/gaussnewton.cpp.o
gaussnewton: CMakeFiles/gaussnewton.dir/build.make
gaussnewton: /usr/local/lib/libopencv_stitching.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_video.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_objdetect.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_photo.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_gapi.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_calib3d.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_ml.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_dnn.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_features2d.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_flann.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_highgui.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_videoio.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_imgproc.so.4.0.0
gaussnewton: /usr/local/lib/libopencv_core.so.4.0.0
gaussnewton: CMakeFiles/gaussnewton.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liuhongwei/workspace/slam/PA56/code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gaussnewton"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gaussnewton.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gaussnewton.dir/build: gaussnewton

.PHONY : CMakeFiles/gaussnewton.dir/build

CMakeFiles/gaussnewton.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gaussnewton.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gaussnewton.dir/clean

CMakeFiles/gaussnewton.dir/depend:
	cd /home/liuhongwei/workspace/slam/PA56/code && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuhongwei/workspace/slam/PA56/code /home/liuhongwei/workspace/slam/PA56/code /home/liuhongwei/workspace/slam/PA56/code /home/liuhongwei/workspace/slam/PA56/code /home/liuhongwei/workspace/slam/PA56/code/CMakeFiles/gaussnewton.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gaussnewton.dir/depend

