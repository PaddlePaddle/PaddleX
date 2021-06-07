cmake_minimum_required(VERSION 3.0)
project(PaddleDeploy CXX C)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
SET(OPENVINO_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(GFLAGS_DIR "" CACHE PATH "Location of libraries")
SET(NGRAPH_LIB "" CACHE PATH "Location of libraries")

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

# Openvino
MESSAGE("${GFLAGS_DIR}")
include_directories("${GFLAGS_DIR}/include")
link_directories("${GFLAGS_DIR}/lib")

include_directories("${GLOG_DIR}/include")
link_directories("${GLOG_DIR}/lib")

if(WIN32)
    set(DEPS ${OPENVINO_DIR}/lib/intel64/Release/inference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/Release/inference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} glog gflags_static libyaml-cppmt)
    else()
    if (${ARCH} STREQUAL "armv7")
        if(WITH_STATIC_LIB)
            set(DEPS ${OPENVINO_DIR}/lib/armv7l/libinference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
        else()
            set(DEPS ${OPENVINO_DIR}/lib/armv7l/libinference_engine${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine_legacy${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif()
    else()
        if(WITH_STATIC_LIB)
            set(DEPS ${OPENVINO_DIR}/lib/intel64/libinference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
        else()
            set(DEPS ${OPENVINO_DIR}/lib/intel64/libinference_engine${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine_legacy${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif()
    endif()
    set(DEPS ${DEPS} glog gflags yaml-cpp)
endif(WIN32)

# OPENCV
if (WIN32)
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
  unset(OpenCV_DIR CACHE)
else ()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
endif ()

set(DEPS ${DEPS} ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

message("-----DEPS = ${DEPS}")

#project
include_directories("${PROJECT_ROOT_DIR}")

aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/common/src SRC)
set(ENGINE_SRC ${PROJECT_ROOT_DIR}/model_deploy/engine/src/openvino_engine.cpp)

#detector seg
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppdet/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppseg/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppclas/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/paddlex/src DETECTOR_SRC)

add_executable(model_infer ${PROJECT_ROOT_DIR}/demo/onnx_openvino/model_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC})
ADD_DEPENDENCIES(model_infer ext-yaml-cpp)
target_link_libraries(model_infer ${DEPS})
