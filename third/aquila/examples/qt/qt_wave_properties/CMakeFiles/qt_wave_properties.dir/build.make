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
include examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/depend.make

# Include the progress variables for this target.
include examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/progress.make

# Include the compile flags for this target's objects.
include examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/flags.make

examples/qt/qt_wave_properties/ui_MainWindow.h: examples/qt/qt_wave_properties/MainWindow.ui
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ui_MainWindow.h"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/lib64/qt5/bin/uic -o /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/ui_MainWindow.h /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/MainWindow.ui

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/flags.make
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o: examples/qt/qt_wave_properties/WavePropertiesWidget.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o -c /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/WavePropertiesWidget.cpp

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.i"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/WavePropertiesWidget.cpp > CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.i

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.s"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/WavePropertiesWidget.cpp -o CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.s

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.requires:

.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.requires

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.provides: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.requires
	$(MAKE) -f examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/build.make examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.provides.build
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.provides

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.provides.build: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o


examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/flags.make
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o: examples/qt/qt_wave_properties/MainWindow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o -c /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/MainWindow.cpp

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.i"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/MainWindow.cpp > CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.i

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.s"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/MainWindow.cpp -o CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.s

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.requires:

.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.requires

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.provides: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.requires
	$(MAKE) -f examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/build.make examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.provides.build
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.provides

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.provides.build: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o


examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/flags.make
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o: examples/qt/qt_wave_properties/qt_wave_properties.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o -c /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/qt_wave_properties.cpp

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.i"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/qt_wave_properties.cpp > CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.i

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.s"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/qt_wave_properties.cpp -o CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.s

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.requires:

.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.requires

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.provides: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.requires
	$(MAKE) -f examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/build.make examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.provides.build
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.provides

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.provides.build: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o


examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/flags.make
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o: examples/qt/qt_wave_properties/qt_wave_properties_automoc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o -c /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/qt_wave_properties_automoc.cpp

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.i"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/qt_wave_properties_automoc.cpp > CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.i

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.s"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/qt_wave_properties_automoc.cpp -o CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.s

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.requires:

.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.requires

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.provides: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.requires
	$(MAKE) -f examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/build.make examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.provides.build
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.provides

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.provides.build: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o


# Object files for target qt_wave_properties
qt_wave_properties_OBJECTS = \
"CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o" \
"CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o" \
"CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o" \
"CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o"

# External object files for target qt_wave_properties
qt_wave_properties_EXTERNAL_OBJECTS =

examples/qt/qt_wave_properties/qt_wave_properties: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o
examples/qt/qt_wave_properties/qt_wave_properties: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o
examples/qt/qt_wave_properties/qt_wave_properties: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o
examples/qt/qt_wave_properties/qt_wave_properties: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o
examples/qt/qt_wave_properties/qt_wave_properties: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/build.make
examples/qt/qt_wave_properties/qt_wave_properties: libAquila.a
examples/qt/qt_wave_properties/qt_wave_properties: /usr/lib64/libQt5Widgets.so.5.6.2
examples/qt/qt_wave_properties/qt_wave_properties: lib/libOoura_fft.a
examples/qt/qt_wave_properties/qt_wave_properties: /usr/lib64/libQt5Gui.so.5.6.2
examples/qt/qt_wave_properties/qt_wave_properties: /usr/lib64/libQt5Core.so.5.6.2
examples/qt/qt_wave_properties/qt_wave_properties: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/elchaschab/devel/PixelShift/third/aquila/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable qt_wave_properties"
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/qt_wave_properties.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/build: examples/qt/qt_wave_properties/qt_wave_properties

.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/build

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/requires: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/WavePropertiesWidget.cpp.o.requires
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/requires: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/MainWindow.cpp.o.requires
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/requires: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties.cpp.o.requires
examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/requires: examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/qt_wave_properties_automoc.cpp.o.requires

.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/requires

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/clean:
	cd /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties && $(CMAKE_COMMAND) -P CMakeFiles/qt_wave_properties.dir/cmake_clean.cmake
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/clean

examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/depend: examples/qt/qt_wave_properties/ui_MainWindow.h
	cd /home/elchaschab/devel/PixelShift/third/aquila && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/elchaschab/devel/PixelShift/third/aquila /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties /home/elchaschab/devel/PixelShift/third/aquila /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties /home/elchaschab/devel/PixelShift/third/aquila/examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/qt/qt_wave_properties/CMakeFiles/qt_wave_properties.dir/depend

