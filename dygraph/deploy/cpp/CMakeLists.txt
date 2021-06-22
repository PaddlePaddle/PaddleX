cmake_minimum_required(VERSION 3.0)
project(PaddleDeploy CXX C)

if (WIN32)
  option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
else()
  option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)
endif()
# Paddle
option(WITH_MKL        "Compile demo with MKL/OpenBlas support,defaultuseMKL."          ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_PADDLE_TENSORRT "Compile demo with TensorRT."   OFF)
# engine
option(WITH_OPENVINO "Compile demo with TensorRT."   OFF)
option(WITH_TENSORRT "Compile demo with TensorRT."   OFF)

SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(DEPS "" CACHE PATH "Location of libraries")
# Paddle
SET(TENSORRT_DIR "" CACHE PATH "Location of libraries")
SET(PADDLE_DIR "" CACHE PATH "Location of libraries")
SET(CUDA_LIB "" CACHE PATH "Location of libraries")
#OpenVINO
SET(GFLAGS_DIR "" CACHE PATH "Location of libraries")
SET(OPENVINO_DIR "" CACHE PATH "Location of libraries")
SET(NGRAPH_LIB "" CACHE PATH "Location of libraries")

if (WIN32)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddle_deploy)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddle_deploy)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddle_deploy)
else()
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demo)
endif()

#project
include_directories("${PROJECT_SOURCE_DIR}")

# common
aux_source_directory(${PROJECT_SOURCE_DIR}/model_deploy/common/src SRC)

# det seg clas pdx src
aux_source_directory(${PROJECT_SOURCE_DIR}/model_deploy/ppdet/src DETECTOR_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/model_deploy/ppseg/src DETECTOR_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/model_deploy/ppclas/src DETECTOR_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/model_deploy/paddlex/src DETECTOR_SRC)

#yaml-cpp
if(WIN32)
  SET(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "yaml build shared library.")
else()
  SET(YAML_BUILD_SHARED_LIBS ON CACHE BOOL "yaml build shared library.")
endif(WIN32)
include(${PROJECT_SOURCE_DIR}/cmake/yaml-cpp.cmake)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")

#OPENCV
if (NOT (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
  if (NOT DEFINED OPENCV_DIR OR ${OPENCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
  endif()
endif()

if (WIN32)
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
  unset(OpenCV_DIR CACHE)
else ()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") # x86_64 aarch64
    set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    file(GLOB OpenCV_LIBS /usr/lib/aarch64-linux-gnu/libopencv_*${CMAKE_SHARED_LIBRARY_SUFFIX})
    message("OpenCV libs: ${OpenCV_LIBS}")
  else()
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
  endif()
endif ()

set(DEPS ${DEPS} ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

# MD
macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} ${OpenMP_C_FLAGS}")
        set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o2 -fopenmp -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

# engine
if(WITH_OPENVINO)
  add_subdirectory(demo/onnx_openvino)
else ()
  add_subdirectory(demo)
endif()

message("-----DEPS = ${DEPS}")
