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

# Utility rule file for qt_wave_properties_automoc.

# Include the progress variables for this target.
include examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/progress.make

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic moc for target qt_wave_properties"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/cmake -E cmake_autogen /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/ ""

qt_wave_properties_automoc: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc
qt_wave_properties_automoc: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/build.make

.PHONY : qt_wave_properties_automoc

# Rule to build all files generated by this target.
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/build: qt_wave_properties_automoc

.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/build

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/clean:
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && $(CMAKE_COMMAND) -P CMakeFiles/qt_wave_properties_automoc.dir/cmake_clean.cmake
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/clean

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/depend:
	cd /home/elchaschab/devel/PixelShift/third/aquila && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/elchaschab/devel/PixelShift/third/aquila /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties /home/elchaschab/devel/PixelShift/third/aquila /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties_automoc.dir/depend
