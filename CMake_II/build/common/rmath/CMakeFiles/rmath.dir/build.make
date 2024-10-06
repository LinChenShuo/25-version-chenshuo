# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build

# Include any dependencies generated for this target.
include common/rmath/CMakeFiles/rmath.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include common/rmath/CMakeFiles/rmath.dir/compiler_depend.make

# Include the progress variables for this target.
include common/rmath/CMakeFiles/rmath.dir/progress.make

# Include the compile flags for this target's objects.
include common/rmath/CMakeFiles/rmath.dir/flags.make

common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.o: common/rmath/CMakeFiles/rmath.dir/flags.make
common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.o: ../common/rmath/src/rmath.cpp
common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.o: common/rmath/CMakeFiles/rmath.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.o"
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.o -MF CMakeFiles/rmath.dir/src/rmath.cpp.o.d -o CMakeFiles/rmath.dir/src/rmath.cpp.o -c /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/common/rmath/src/rmath.cpp

common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rmath.dir/src/rmath.cpp.i"
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/common/rmath/src/rmath.cpp > CMakeFiles/rmath.dir/src/rmath.cpp.i

common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rmath.dir/src/rmath.cpp.s"
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/common/rmath/src/rmath.cpp -o CMakeFiles/rmath.dir/src/rmath.cpp.s

# Object files for target rmath
rmath_OBJECTS = \
"CMakeFiles/rmath.dir/src/rmath.cpp.o"

# External object files for target rmath
rmath_EXTERNAL_OBJECTS =

common/rmath/librmath.a: common/rmath/CMakeFiles/rmath.dir/src/rmath.cpp.o
common/rmath/librmath.a: common/rmath/CMakeFiles/rmath.dir/build.make
common/rmath/librmath.a: common/rmath/CMakeFiles/rmath.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library librmath.a"
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath && $(CMAKE_COMMAND) -P CMakeFiles/rmath.dir/cmake_clean_target.cmake
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rmath.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
common/rmath/CMakeFiles/rmath.dir/build: common/rmath/librmath.a
.PHONY : common/rmath/CMakeFiles/rmath.dir/build

common/rmath/CMakeFiles/rmath.dir/clean:
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath && $(CMAKE_COMMAND) -P CMakeFiles/rmath.dir/cmake_clean.cmake
.PHONY : common/rmath/CMakeFiles/rmath.dir/clean

common/rmath/CMakeFiles/rmath.dir/depend:
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/common/rmath /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/common/rmath/CMakeFiles/rmath.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : common/rmath/CMakeFiles/rmath.dir/depend

