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
include CMakeFiles/client.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/client.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/client.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/client.dir/flags.make

CMakeFiles/client.dir/client.cpp.o: CMakeFiles/client.dir/flags.make
CMakeFiles/client.dir/client.cpp.o: ../client.cpp
CMakeFiles/client.dir/client.cpp.o: CMakeFiles/client.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/client.dir/client.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/client.dir/client.cpp.o -MF CMakeFiles/client.dir/client.cpp.o.d -o CMakeFiles/client.dir/client.cpp.o -c /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/client.cpp

CMakeFiles/client.dir/client.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/client.dir/client.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/client.cpp > CMakeFiles/client.dir/client.cpp.i

CMakeFiles/client.dir/client.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/client.dir/client.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/client.cpp -o CMakeFiles/client.dir/client.cpp.s

# Object files for target client
client_OBJECTS = \
"CMakeFiles/client.dir/client.cpp.o"

# External object files for target client
client_EXTERNAL_OBJECTS =

client: CMakeFiles/client.dir/client.cpp.o
client: CMakeFiles/client.dir/build.make
client: common/rmath/librmath.a
client: modules/assembly1/libassembly1.a
client: modules/assembly2/libassembly2.a
client: modules/module1/libmodule1.a
client: modules/module2/libmodule2.a
client: modules/assembly2/libassembly2.a
client: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
client: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
client: CMakeFiles/client.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable client"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/client.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/client.dir/build: client
.PHONY : CMakeFiles/client.dir/build

CMakeFiles/client.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/client.dir/cmake_clean.cmake
.PHONY : CMakeFiles/client.dir/clean

CMakeFiles/client.dir/depend:
	cd /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build /home/shuo/Desktop/vision/25-vision-chen-shuo/CMake_II/build/CMakeFiles/client.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/client.dir/depend

