# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/elchaschab/devel/PixelShift/third/aquila

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/elchaschab/devel/PixelShift/third/aquila

# Include any dependencies generated for this target.
include examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/depend.make

# Include the progress variables for this target.
include examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/progress.make

# Include the compile flags for this target's objects.
include examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/flags.make

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o: examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/flags.make
examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o: examples/mfcc_calculation/mfcc_calculation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o -c /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation/mfcc_calculation.cpp

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.i"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation/mfcc_calculation.cpp > CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.i

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.s"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation/mfcc_calculation.cpp -o CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.s

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.requires:

.PHONY : examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.requires

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.provides: examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.requires
	$(MAKE) -f examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/build.make examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.provides.build
.PHONY : examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.provides

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.provides.build: examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o


# Object files for target mfcc_calculation
mfcc_calculation_OBJECTS = \
"CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o"

# External object files for target mfcc_calculation
mfcc_calculation_EXTERNAL_OBJECTS =

examples/mfcc_calculation/mfcc_calculation: examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o
examples/mfcc_calculation/mfcc_calculation: examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/build.make
examples/mfcc_calculation/mfcc_calculation: libAquila.a
examples/mfcc_calculation/mfcc_calculation: lib/libOoura_fft.a
examples/mfcc_calculation/mfcc_calculation: examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mfcc_calculation"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mfcc_calculation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/build: examples/mfcc_calculation/mfcc_calculation

.PHONY : examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/build

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/requires: examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/mfcc_calculation.cpp.o.requires

.PHONY : examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/requires

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/clean:
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation && $(CMAKE_COMMAND) -P CMakeFiles/mfcc_calculation.dir/cmake_clean.cmake
.PHONY : examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/clean

examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/depend:
	cd /home/elchaschab/devel/PixelShift/third/aquila && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/elchaschab/devel/PixelShift/third/aquila /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation /home/elchaschab/devel/PixelShift/third/aquila /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation /home/elchaschab/devel/PixelShift/third/aquila/examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/mfcc_calculation/CMakeFiles/mfcc_calculation.dir/depend

