# created from template OpenCV included c++ project
#   by BreakDS 
#   Mar 23 2012 @ Univeristy of Wisconsin-Madison
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Ask CMake to search for current c/c++ compiler
set(CMAKE_CXX_COMPILER_INIT "c++")


# Project Description Starts Here
PROJECT(Semi)


# System & Project Information
MESSAGE(STATUS "Project Name:           ${PROJECT_NAME}")
MESSAGE(STATUS "Project Source Dir:     ${PROJECT_SOURCE_DIR}")
MESSAGE(STATUS "Kernel:                 ${CMAKE_SYSTEM}")
if ("$ENV{HOST}" MATCHES "")
  MESSAGE(STATUS "Hostname:               $ENV{HOSTNAME}")
  MESSAGE(STATUS "Distribution:           Debian GNU/Linux")
else("$ENV{HOST}" MATCHES "")
  MESSAGE(STATUS "Host:                   $ENV{HOST}")
endif("$ENV{HOST}" MATCHES "")
MESSAGE(STATUS "Build Type:             ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "GCC:                    ${CMAKE_CXX_COMPILER}")


# Include Directory Structure for breakds@emperor
if ("$ENV{HOST}" MATCHES "emperor.cs.wisc.edu")
  include_directories(
    "/scratch/usr/local/include"
    "~/pf/projects"
    ".")
  # OpenCV Library
  add_library( OPENCV SHARED IMPORTED)
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_core.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_highgui.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_features2d.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_objdetect.so" )
  MESSAGE(STATUS "emperor.cs.wisc.edu specific configuration on!")
endif ("$ENV{HOST}" MATCHES "emperor.cs.wisc.edu")


# Include Directory Structure for bangel@beggar
if ("$ENV{HOST}" MATCHES "beggar.cs.wisc.edu")
  include_directories(
    "/scratch/opt/include"
    ".")
  # OpenCV Library
  add_library( OPENCV SHARED IMPORTED)
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_core.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_highgui.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_features2d.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib//libopencv_objdetect.so" )
  MESSAGE(STATUS "beggar.cs.wisc.edu specific configuration on!")
endif ("$ENV{HOST}" MATCHES "beggar.cs.wisc.edu")



# Include Directory Structure for lizy@tarzan
if ("$ENV{HOST}" MATCHES "tarzan.cs.wisc.edu")
  include_directories(
    "/u/l/i/lizy/myinstall/include"
    ".")
  # OpenCV Library
  add_library( OPENCV SHARED IMPORTED)
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/u/l/i/lizy/myinstall/lib/libopencv_core.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/u/l/i/lizy/myinstall/lib/libopencv_highgui.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/u/l/i/lizy/myinstall/lib/libopencv_features2d.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/u/l/i/lizy/myinstall/lib/libopencv_objdetect.so" )
  MESSAGE(STATUS "tarzan.cs.wisc.edu specific configuration on!")
endif ("$ENV{HOST}" MATCHES "tarzan.cs.wisc.edu")

# For breakds@{any debian box}
if ("$ENV{HOST}" MATCHES "")
  include_directories(
    "/scratch/usr/local/include"
    "~/pf/projects"
    ".")
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
endif ("$ENV{HOST}" MATCHES "")



if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)



# Whether build as shared libraries by default
# SET(BUILD_SHARED_LIBS ON)

# LINKER_FLAGS
# set( CMAKE_EXE_LINKER_FLAGS "-lXext -lXft -lfontconfig -lXinerama -lpthread -ldl -lm -lX11" )


# COMPILATION_FLAGS
# Note -Wno-non-virtual-dtor is for a defect in OpenCV
set(CMAKE_CXX_FLAGS "${OPENCV_FLAGS} -Wall -Wextra -Wno-non-virtual-dtor -std=c++0x")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g  -D_GLIBCXX_DEBUG")
set(CMAKE_CXX_FLAGS_GPROF "-O2 -pg")


# FeatureGen
ADD_EXECUTABLE(PatchShift sample/PatchShift.cpp)
# TARGET_LINK_LIBRARIES(PatchShift OPENCV)
