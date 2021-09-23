# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /easy_data/easy_perception/easy_arm/ambarella

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /easy_data/easy_perception/easy_arm/ambarella/build

# Include any dependencies generated for this target.
include cnn_runtime/tests/CMakeFiles/posenet_test.dir/depend.make

# Include the progress variables for this target.
include cnn_runtime/tests/CMakeFiles/posenet_test.dir/progress.make

# Include the compile flags for this target's objects.
include cnn_runtime/tests/CMakeFiles/posenet_test.dir/flags.make

cnn_runtime/tests/CMakeFiles/posenet_test.dir/posenet_test.cpp.o: cnn_runtime/tests/CMakeFiles/posenet_test.dir/flags.make
cnn_runtime/tests/CMakeFiles/posenet_test.dir/posenet_test.cpp.o: ../cnn_runtime/tests/posenet_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/easy_data/easy_perception/easy_arm/ambarella/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cnn_runtime/tests/CMakeFiles/posenet_test.dir/posenet_test.cpp.o"
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && /usr/local/linaro-aarch64-2018.08-gcc8.2/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/posenet_test.dir/posenet_test.cpp.o -c /easy_data/easy_perception/easy_arm/ambarella/cnn_runtime/tests/posenet_test.cpp

cnn_runtime/tests/CMakeFiles/posenet_test.dir/posenet_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/posenet_test.dir/posenet_test.cpp.i"
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && /usr/local/linaro-aarch64-2018.08-gcc8.2/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /easy_data/easy_perception/easy_arm/ambarella/cnn_runtime/tests/posenet_test.cpp > CMakeFiles/posenet_test.dir/posenet_test.cpp.i

cnn_runtime/tests/CMakeFiles/posenet_test.dir/posenet_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/posenet_test.dir/posenet_test.cpp.s"
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && /usr/local/linaro-aarch64-2018.08-gcc8.2/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /easy_data/easy_perception/easy_arm/ambarella/cnn_runtime/tests/posenet_test.cpp -o CMakeFiles/posenet_test.dir/posenet_test.cpp.s

cnn_runtime/tests/CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.o: cnn_runtime/tests/CMakeFiles/posenet_test.dir/flags.make
cnn_runtime/tests/CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.o: ../utility/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/easy_data/easy_perception/easy_arm/ambarella/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object cnn_runtime/tests/CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.o"
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && /usr/local/linaro-aarch64-2018.08-gcc8.2/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.o -c /easy_data/easy_perception/easy_arm/ambarella/utility/utils.cpp

cnn_runtime/tests/CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.i"
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && /usr/local/linaro-aarch64-2018.08-gcc8.2/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /easy_data/easy_perception/easy_arm/ambarella/utility/utils.cpp > CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.i

cnn_runtime/tests/CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.s"
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && /usr/local/linaro-aarch64-2018.08-gcc8.2/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /easy_data/easy_perception/easy_arm/ambarella/utility/utils.cpp -o CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.s

# Object files for target posenet_test
posenet_test_OBJECTS = \
"CMakeFiles/posenet_test.dir/posenet_test.cpp.o" \
"CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.o"

# External object files for target posenet_test
posenet_test_EXTERNAL_OBJECTS =

posenet_test: cnn_runtime/tests/CMakeFiles/posenet_test.dir/posenet_test.cpp.o
posenet_test: cnn_runtime/tests/CMakeFiles/posenet_test.dir/__/__/utility/utils.cpp.o
posenet_test: cnn_runtime/tests/CMakeFiles/posenet_test.dir/build.make
posenet_test: libcnn_runtime.so
posenet_test: cnn_runtime/tests/CMakeFiles/posenet_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/easy_data/easy_perception/easy_arm/ambarella/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../posenet_test"
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/posenet_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cnn_runtime/tests/CMakeFiles/posenet_test.dir/build: posenet_test

.PHONY : cnn_runtime/tests/CMakeFiles/posenet_test.dir/build

cnn_runtime/tests/CMakeFiles/posenet_test.dir/clean:
	cd /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests && $(CMAKE_COMMAND) -P CMakeFiles/posenet_test.dir/cmake_clean.cmake
.PHONY : cnn_runtime/tests/CMakeFiles/posenet_test.dir/clean

cnn_runtime/tests/CMakeFiles/posenet_test.dir/depend:
	cd /easy_data/easy_perception/easy_arm/ambarella/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /easy_data/easy_perception/easy_arm/ambarella /easy_data/easy_perception/easy_arm/ambarella/cnn_runtime/tests /easy_data/easy_perception/easy_arm/ambarella/build /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests /easy_data/easy_perception/easy_arm/ambarella/build/cnn_runtime/tests/CMakeFiles/posenet_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cnn_runtime/tests/CMakeFiles/posenet_test.dir/depend

