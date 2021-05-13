cmake_minimum_required(VERSION 3.0)
project(PaddleDeploy CXX C)

SET(TRITON_CLIENT "" CACHE PATH "Location of libraries")
SET(GLOG_DIR "" CACHE PATH "Location of libraries")
SET(GFLAGS_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")

SET(PROJECT_ROOT_DIR  "../../" CACHE PATH  "root directory of project.")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o2 -fopenmp -std=c++11")
#source
include_directories("${CMAKE_SOURCE_DIR}/")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")

# yaml-cpp
SET(YAML_BUILD_SHARED_LIBS ON CACHE BOOL "yaml build shared library.")

include(${PROJECT_ROOT_DIR}/cmake/yaml-cpp.cmake)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")

# Triton
MESSAGE("${GFLAGS_DIR}")
include_directories("${GFLAGS_DIR}/include")
link_directories("${GFLAGS_DIR}/lib")

include_directories("${GLOG_DIR}/include")
link_directories("${GLOG_DIR}/lib")

include_directories("${TRITON_CLIENT}/include")
link_directories("${TRITON_CLIENT}/lib")

set(DEPS ${TRITON_CLIENT}/lib/libhttpclient${CMAKE_SHARED_LIBRARY_SUFFIX})
set(DEPS ${DEPS} glog gflags protobuf yaml-cpp)

# OPENCV
if (NOT (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
  if (NOT DEFINED OPENCV_DIR OR ${OPENCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
  endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") # x86_64 aarch64
  set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
  file(GLOB OpenCV_LIBS /usr/lib/aarch64-linux-gnu/libopencv_*${CMAKE_SHARED_LIBRARY_SUFFIX})
  message("OpenCV libs: ${OpenCV_LIBS}")
else()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
endif()

set(DEPS ${DEPS} ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

message("-----DEPS = ${DEPS}")

#project
include_directories("${PROJECT_ROOT_DIR}")

aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/common/src SRC)
set(ENGINE_SRC ${PROJECT_ROOT_DIR}/model_deploy/engine/src/triton_engine.cpp)

#detector seg
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppdet/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppseg/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppclas/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/paddlex/src DETECTOR_SRC)

add_executable(model_infer ${PROJECT_ROOT_DIR}/demo/triton/model_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC})
ADD_DEPENDENCIES(model_infer ext-yaml-cpp)
target_link_libraries(model_infer ${DEPS})
