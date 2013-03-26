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

option (BUILD_UNITTEST "build executables in purpose of unittest." ON)


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

  # CUDA SDK 5.0
  set(CUDA_TOOLKIT_ROOT_DIR "/s/cuda-5.0.35/") # Force it to search for CUDA 4.2.9
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_20")
endif ("${HOSTNAME}" MATCHES "emperor.cs.wisc.edu")


# ================================================================================
# Include Directory Structure for breakds@beggar
if ("${HOSTNAME}" MATCHES "beggar.cs.wisc.edu")
  include_directories(
    "/scratch/usr/local/include"
    "~/pf/projects"
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
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_objdetect.so" )
  set(OPENCV "OPENCV")

  # CUDA SDK 5.0
  set(CUDA_TOOLKIT_ROOT_DIR "/s/cuda-5.0.35/") # Force it to search for CUDA 4.2.9
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_30")
endif ("${HOSTNAME}" MATCHES "beggar.cs.wisc.edu")




# ================================================================================
# Include Directory Structure for break@zhang-submit.chtc.wisc.edu
if ("${HOSTNAME}" MATCHES "zhang-submit.chtc.wisc.edu")
  include_directories(
    "~/pf/projects"
    ".")
  # OpenCV Library
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
  set(OPENCV "")
  
  # CUDA SDK 5.0
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_30 --compiler-bindir /usr/bin")
endif ("${HOSTNAME}" MATCHES "zhang-submit.chtc.wisc.edu")

# ================================================================================
# Include Directory Structure for break@zhang-submit.chtc.wisc.edu
if ("${HOSTNAME}" MATCHES "zhang-c001.chtc.wisc.edu")
  include_directories(
    "~/pf/projects"
    ".")
  # OpenCV Library
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
  set(OPENCV "")
  
  # CUDA SDK 5.0
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_30 --compiler-bindir /usr/bin")
endif ("${HOSTNAME}" MATCHES "zhang-c001.chtc.wisc.edu")



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




# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
# cuda_add_library(cuda_opt SHARED opt/BP_CUDA.cu)




# ================================================================================
# COMPILATION_FLAGS
# Note -Wno-non-virtual-dtor is for a defect in OpenCV
set(CMAKE_CXX_FLAGS "${OPENCV_FLAGS} -Wall -Wextra -Wno-non-virtual-dtor -std=c++0x")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fopenmp")
set(CMAKE_CXX_FLAGS_GPROF "-O1 -pg")


if (BUILD_UNITTEST)
  ADD_EXECUTABLE(TestFeatImage data/UnitTest/FeatImage.cpp)
  TARGET_LINK_LIBRARIES(TestFeatImage ${OPENCV})
  ADD_EXECUTABLE(TestRotation data/UnitTest/RotationTest.cpp)
  TARGET_LINK_LIBRARIES(TestRotation ${OPENCV})
  ADD_EXECUTABLE(TestScale data/UnitTest/ScaleTest.cpp)
  TARGET_LINK_LIBRARIES(TestScale ${OPENCV})
  ADD_EXECUTABLE(icon interfaces/UnitTest/IconList.cpp)
  TARGET_LINK_LIBRARIES(icon ${OPENCV})
  ADD_EXECUTABLE(leaf query/UnitTest/LeafView.cpp)
  TARGET_LINK_LIBRARIES(leaf ${OPENCV})
endif (BUILD_UNITTEST)

# ADD_EXECUTABLE(code experiment/code_test.cpp)
# TARGET_LINK_LIBRARIES(code ${OPENCV})


# ADD_EXECUTABLE(training experiment/Training.cpp)
# TARGET_LINK_LIBRARIES(training ${OPENCV})

# ADD_EXECUTABLE(testing experiment/Testing.cpp)
# TARGET_LINK_LIBRARIES(testing ${OPENCV})

# ADD_EXECUTABLE(learning experiment/Learning.cpp)
# TARGET_LINK_LIBRARIES(learning ${OPENCV})

# ADD_EXECUTABLE(TrainLabel experiment/TrainLabel.cpp)
# TARGET_LINK_LIBRARIES(TrainLabel ${OPENCV})

# ADD_EXECUTABLE(Vote experiment/Vote.cpp)
# TARGET_LINK_LIBRARIES(Vote ${OPENCV})

# ADD_EXECUTABLE(verify experiment/Verify.cpp)
# TARGET_LINK_LIBRARIES(verify ${OPENCV})

# ADD_EXECUTABLE(forestExp experiment/ForestExp.cpp)
# TARGET_LINK_LIBRARIES(forestExp ${OPENCV})

# ADD_EXECUTABLE(treeQ experiment/TreeQuality.cpp)
# TARGET_LINK_LIBRARIES(treeQ ${OPENCV})

ADD_EXECUTABLE(hpatch sample/HeavyPatch.cpp)
TARGET_LINK_LIBRARIES(hpatch ${OPENCV})









