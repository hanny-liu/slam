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
CMAKE_SOURCE_DIR = /home/liuhongwei/workspace/slam/PA4/answer_7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuhongwei/workspace/slam/PA4/answer_7/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/answer_7.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/answer_7.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/answer_7.dir/flags.make

CMakeFiles/answer_7.dir/main.cpp.o: CMakeFiles/answer_7.dir/flags.make
CMakeFiles/answer_7.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liuhongwei/workspace/slam/PA4/answer_7/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/answer_7.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/answer_7.dir/main.cpp.o -c /home/liuhongwei/workspace/slam/PA4/answer_7/main.cpp

CMakeFiles/answer_7.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/answer_7.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liuhongwei/workspace/slam/PA4/answer_7/main.cpp > CMakeFiles/answer_7.dir/main.cpp.i

CMakeFiles/answer_7.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/answer_7.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liuhongwei/workspace/slam/PA4/answer_7/main.cpp -o CMakeFiles/answer_7.dir/main.cpp.s

# Object files for target answer_7
answer_7_OBJECTS = \
"CMakeFiles/answer_7.dir/main.cpp.o"

# External object files for target answer_7
answer_7_EXTERNAL_OBJECTS =

../runtime/answer_7: CMakeFiles/answer_7.dir/main.cpp.o
../runtime/answer_7: CMakeFiles/answer_7.dir/build.make
../runtime/answer_7: /home/liuhongwei/ORB_SLAM2/Pangolin/build/src/libpangolin.so
../runtime/answer_7: /home/liuhongwei/workspace/slam/slambook/3rdparty/Sophus/build/libSophus.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libGLU.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libGL.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libGLEW.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libSM.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libICE.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libX11.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libXext.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libdc1394.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libavcodec.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libavformat.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libavutil.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libswscale.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libavdevice.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libpng.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libz.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libjpeg.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libtiff.so
../runtime/answer_7: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../runtime/answer_7: CMakeFiles/answer_7.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liuhongwei/workspace/slam/PA4/answer_7/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../runtime/answer_7"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/answer_7.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/answer_7.dir/build: ../runtime/answer_7

.PHONY : CMakeFiles/answer_7.dir/build

CMakeFiles/answer_7.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/answer_7.dir/cmake_clean.cmake
.PHONY : CMakeFiles/answer_7.dir/clean

CMakeFiles/answer_7.dir/depend:
	cd /home/liuhongwei/workspace/slam/PA4/answer_7/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuhongwei/workspace/slam/PA4/answer_7 /home/liuhongwei/workspace/slam/PA4/answer_7 /home/liuhongwei/workspace/slam/PA4/answer_7/cmake-build-debug /home/liuhongwei/workspace/slam/PA4/answer_7/cmake-build-debug /home/liuhongwei/workspace/slam/PA4/answer_7/cmake-build-debug/CMakeFiles/answer_7.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/answer_7.dir/depend

