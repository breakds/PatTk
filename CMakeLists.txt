
# created from template OpenCV included c++ project
#   by BreakDS 
#   Mar 23 2012 @ Univeristy of Wisconsin-Madison
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Ask CMake to search for current c/c++ compiler
set(CMAKE_CXX_COMPILER_INIT "c++")


# Project Description Starts Here
PROJECT(PatTk)


# System & Project Information
MESSAGE(STATUS "Project Name:           ${PROJECT_NAME}")
MESSAGE(STATUS "Project Source Dir:     ${PROJECT_SOURCE_DIR}")
MESSAGE(STATUS "Kernel:                 ${CMAKE_SYSTEM}")
EXECUTE_PROCESS(COMMAND hostname OUTPUT_VARIABLE HOSTNAME)
MESSAGE(STATUS "Hostname:               ${HOSTNAME}")
MESSAGE(STATUS "Build Type:             ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "GCC:                    ${CMAKE_CXX_COMPILER}")


# ================================================================================
# Include Directory Structure for breakds@emperor
if ("${HOSTNAME}" MATCHES "emperor.cs.wisc.edu")
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
  set(OPENCV "OPENCV")

  # CUDA SDK 4.2
  set(CUDA_TOOLKIT_ROOT_DIR "/s/cuda-4.2.9/") # Force it to search for CUDA 4.2.9
  find_package("CUDA" 4.2 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_20")
endif ("${HOSTNAME}" MATCHES "emperor.cs.wisc.edu")






# ================================================================================
# For breakds@AzraelWaker
if ("${HOSTNAME}" MATCHES "AzraelWaker")
  include_directories(
    "~/pf/projects"
    ".")
  
  # Linker Flags
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
  set(OPENCV "")

  # CUDA SDK 5.0
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_30")
endif ("${HOSTNAME}" MATCHES "AzraelWaker")


if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)



# Whether build as shared libraries by default
# SET(BUILD_SHARED_LIBS ON)



# ================================================================================
# Compile CUDA related libraries
MESSAGE(STATUS "CUDA VERISON: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" )
MESSAGE(STATUS "NVCC VERISON: ${CUDA_NVCC_EXECUTABLE}" )




set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
cuda_add_library(cuda_opt SHARED opt/BP_CUDA.cu)




# ================================================================================
# COMPILATION_FLAGS
# Note -Wno-non-virtual-dtor is for a defect in OpenCV
set(CMAKE_CXX_FLAGS "${OPENCV_FLAGS} -Wall -Wextra -Wno-non-virtual-dtor -std=c++0x")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g  -D_GLIBCXX_DEBUG")
set(CMAKE_CXX_FLAGS_GPROF "-O2 -pg")


# PatchShift
# ADD_EXECUTABLE(PatchShift sample/PatchShift.cpp)
# TARGET_LINK_LIBRARIES(PatchShift OPENCV)

# Usage
# ADD_EXECUTABLE(usage sample/usage.cpp)
# TARGET_LINK_LIBRARIES(usage OPENCV)

# AlbumTest
# ADD_EXECUTABLE(AlbumTest sample/AlbumTest.cpp)
# TARGET_LINK_LIBRARIES(AlbumTest OPENCV)

# TreeBuild
# ADD_EXECUTABLE(TreeBuild sample/TreeBuild.cpp)
# TARGET_LINK_LIBRARIES(TreeBuild OPENCV)

# Fake Label Test
# ADD_EXECUTABLE(FakeLabel sample/FakeLabel.cpp)
# TARGET_LINK_LIBRARIES(FakeLabel OPENCV)


# Heavy Patch Test
# ADD_EXECUTABLE(HeavyPatch sample/HeavyPatch.cpp)
# TARGET_LINK_LIBRARIES(HeavyPatch OPENCV)

# BP Test
# ADD_EXECUTABLE(matting opt/UnitTest/matting.cpp)
# TARGET_LINK_LIBRARIES(matting OPENCV cuda_opt)

# IconList Test
# ADD_EXECUTABLE(IconList interfaces/UnitTest/IconList.cpp)
# TARGET_LINK_LIBRARIES(IconList OPENCV)

# Stiching
# ADD_EXECUTABLE(Stitching sample/Stitching.cpp)
# TARGET_LINK_LIBRARIES(Stitching OPENCV)

# Graph Builder
ADD_EXECUTABLE(builder sample/GraphBuilder.cpp)
TARGET_LINK_LIBRARIES(builder ${OPENCV} cuda_opt)

# Stitching Result
ADD_EXECUTABLE(Stitching sample/Stitching.cpp)
TARGET_LINK_LIBRARIES(Stitching ${OPENCV})


