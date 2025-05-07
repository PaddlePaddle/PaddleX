# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
include(ExternalProject)

# The priority strategy for Paddle inference is as follows:
# PADDLEINFERENCE_DIRECTORY > custom PADDLEINFERENCE_URL > default PADDLEINFERENCE_URL.

if(WITH_GPU AND WITH_IPU)
  message(FATAL_ERROR "Cannot build with WITH_GPU=ON and WITH_IPU=ON on the same time.")
endif()

# Custom options for Paddle Inference backend
option(PADDLEINFERENCE_DIRECTORY "Directory of custom Paddle Inference library" OFF)
option(PADDLEINFERENCE_API_CUSTOM_OP "Whether building with custom paddle ops" OFF)
option(PADDLEINFERENCE_API_COMPAT_2_4_x "Whether using Paddle Inference 2.4.x" OFF)
option(PADDLEINFERENCE_API_COMPAT_2_5_x "Whether using Paddle Inference 2.5.x" OFF)
option(PADDLEINFERENCE_API_COMPAT_2_6_x "Whether using Paddle Inference 2.6.x" OFF)
option(PADDLEINFERENCE_API_COMPAT_DEV "Whether using Paddle Inference latest dev" OFF)
option(PADDLEINFERENCE_API_COMPAT_CUDA_SM_80 "Whether using Paddle Inference with CUDA sm_80(A100)" OFF)

set(PADDLEINFERENCE_URL "" CACHE STRING "URL of the custom Paddle Inference library")
set(PADDLEINFERENCE_VERSION "" CACHE STRING "Paddle Inference version")

set(PADDLEINFERENCE_PROJECT "extern_paddle_inference")
set(PADDLEINFERENCE_PREFIX_DIR ${THIRD_PARTY_PATH}/paddle_inference)
set(PADDLEINFERENCE_SOURCE_DIR
    ${THIRD_PARTY_PATH}/paddle_inference/src/${PADDLEINFERENCE_PROJECT})
set(PADDLEINFERENCE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/paddle_inference)

set(PADDLEINFERENCE_INC_DIR "${PADDLEINFERENCE_INSTALL_DIR}"
    CACHE PATH "paddle_inference include directory." FORCE)    
set(PADDLEINFERENCE_LIB_DIR
    "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/"
    CACHE PATH "paddle_inference lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${PADDLEINFERENCE_LIB_DIR}")

if(PADDLEINFERENCE_DIRECTORY)
  set(PADDLEINFERENCE_INC_DIR ${PADDLEINFERENCE_DIRECTORY})
endif()

include_directories(${PADDLEINFERENCE_INC_DIR})

if(PADDLEINFERENCE_DIRECTORY)
  # Use custom Paddle Inference libs.
  if(EXISTS "${THIRD_PARTY_PATH}/install/paddle_inference")
    file(REMOVE_RECURSE "${THIRD_PARTY_PATH}/install/paddle_inference")
  endif()
  if(NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
  endif()  
  message(STATUS "Copying ${PADDLEINFERENCE_DIRECTORY} to ${THIRD_PARTY_PATH}/install/paddle_inference ...")
  if(WIN32)
    execute_process(COMMAND mkdir -p ${THIRD_PARTY_PATH}/install)
    execute_process(COMMAND cp -r ${PADDLEINFERENCE_DIRECTORY} ${THIRD_PARTY_PATH}/install/paddle_inference)
  else()
    execute_process(COMMAND mkdir -p ${THIRD_PARTY_PATH}/install)
    execute_process(COMMAND cp -r ${PADDLEINFERENCE_DIRECTORY} ${THIRD_PARTY_PATH}/install/paddle_inference)
    execute_process(COMMAND rm -rf ${THIRD_PARTY_PATH}/install/paddle_inference/paddle/lib/*.a)
  endif()
else()

  # Custom Paddle Inference URL
  if (NOT PADDLEINFERENCE_URL)

    # Use default Paddle Inference libs.
    set(PADDLEINFERENCE_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
    if(WIN32)
      if (WITH_GPU)
        set(PADDLEINFERENCE_FILE "paddle_inference-win-x64-gpu-trt8.5.2.2-mkl-2.5.0.281761089e.zip")
        set(PADDLEINFERENCE_VERSION "2.5.0.281761089e")
      else()
        set(PADDLEINFERENCE_FILE "paddle_inference-win-x64-mkl-2.5.0.281761089e.zip")
        set(PADDLEINFERENCE_VERSION "2.5.0.281761089e")
      endif()
    elseif(APPLE)
      if(CURRENT_OSX_ARCH MATCHES "arm64")
        message(FATAL_ERROR "Paddle Backend doesn't support Mac OSX with Arm64 now.")
        set(PADDLEINFERENCE_FILE "paddle_inference-osx-arm64-openblas-0.0.0.660f781b77.tgz")
      else()
        # TODO(qiuyanjun): Should remove this old paddle inference lib
        # set(PADDLEINFERENCE_FILE "paddle_inference-osx-x86_64-2.4-dev3.tgz")
        set(PADDLEINFERENCE_FILE "paddle_inference-osx-x86_64-openblas-0.0.0.660f781b77.tgz")
      endif()
      set(PADDLEINFERENCE_VERSION "0.0.0.660f781b77")
    else()
      # Linux with x86/aarch64 CPU/Arm CPU/GPU/IPU ...
      if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        message(FATAL_ERROR "Paddle Backend doesn't support linux aarch64 now.")
      else()
        # x86_64
        if(WITH_GPU)
          if(PADDLEINFERENCE_API_COMPAT_CUDA_SM_80)
            set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-gpu-trt8.5.2.2-mkl-sm70.sm75.sm80.sm86.nodist-2.5.1.tgz")
            set(PADDLEINFERENCE_VERSION "2.5.1")
          else()
            set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-gpu-trt8.5.2.2-mkl-sm61.sm70.sm75.sm86.nodist-2.5.1.tgz")
            set(PADDLEINFERENCE_VERSION "2.5.1")
          endif()
        else()
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-mkl-2.5.1.tgz")
          set(PADDLEINFERENCE_VERSION "2.5.1")
        endif()
        if(WITH_IPU)
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-ipu-2.4-dev1.tgz")
          # TODO(qiuyanjun): Should use the commit id to tag the version
          set(PADDLEINFERENCE_VERSION "2.4-dev1")
        endif()
        if(WITH_KUNLUNXIN)
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-xpu-openblas-0.0.0.021fd73536.tgz")
          set(PADDLEINFERENCE_VERSION "0.0.0.021fd73536")
        endif()

        if(NEED_ABI0)
          if(WITH_GPU OR WITH_IPU OR WITH_KUNLUNXIN)
            message(WARNING "While NEED_ABI0=ON, only support CPU now, will fallback to CPU.")
          endif()
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-2.4.0-abi0.tgz")
          set(PADDLEINFERENCE_VERSION "2.4.0-abi0")
        endif()
      endif()
    endif()
    set(PADDLEINFERENCE_URL "${PADDLEINFERENCE_URL_BASE}${PADDLEINFERENCE_FILE}")

  endif(PADDLEINFERENCE_URL)
  
  ExternalProject_Add(
    ${PADDLEINFERENCE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLEINFERENCE_URL}
    PREFIX ${PADDLEINFERENCE_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
  	${CMAKE_COMMAND} -E copy_directory ${PADDLEINFERENCE_SOURCE_DIR} ${PADDLEINFERENCE_INSTALL_DIR}
    BUILD_BYPRODUCTS ${PADDLEINFERENCE_COMPILE_LIB})

endif(PADDLEINFERENCE_DIRECTORY)

if (PADDLEINFERENCE_VERSION STREQUAL "")
  message(FATAL_ERROR "The Paddle Inference version is unspecified and cannot be determined.")
endif()
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${PADDLEINFERENCE_VERSION}")
set(PADDLEINFERENCE_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(PADDLEINFERENCE_VERSION_MINOR "${CMAKE_MATCH_2}")
set(PADDLEINFERENCE_VERSION_PATCH "${CMAKE_MATCH_3}")
add_definitions("-DPADDLEINFERENCE_VERSION_MAJOR=${PADDLEINFERENCE_VERSION_MAJOR}")
add_definitions("-DPADDLEINFERENCE_VERSION_MINOR=${PADDLEINFERENCE_VERSION_MINOR}")
add_definitions("-DPADDLEINFERENCE_VERSION_PATCH=${PADDLEINFERENCE_VERSION_PATCH}")

# check libs
if(WIN32)
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/paddle_inference.lib"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  if(PADDLEINFERENCE_VERSION_MAJOR EQUAL 2)
    set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mkldnn/lib/mkldnn.lib")
  else()
    set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onednn/lib/dnnl.lib")
  endif()
  set(OMP_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mklml/lib/libiomp5md.lib")
  set(P2O_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/paddle2onnx/lib/paddle2onnx.lib")
  set(ORT_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onnxruntime/lib/onnxruntime.lib")
elseif(APPLE)
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.dylib"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  if(PADDLEINFERENCE_VERSION_MAJOR EQUAL 2)
    set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mkldnn/lib/libdnnl.so.2")
  else()
    set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onednn/lib/libdnnl.so.3")
  endif()
  set(OMP_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mklml/lib/libiomp5.so")
  set(P2O_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/paddle2onnx/lib/libpaddle2onnx.dylib")
  set(ORT_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onnxruntime/lib/libonnxruntime.dylib")
else()
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.so"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  if(PADDLEINFERENCE_VERSION_MAJOR EQUAL 2)
    set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mkldnn/lib/libdnnl.so.2")
  else()
    set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onednn/lib/libdnnl.so.3")
  endif()
  set(OMP_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mklml/lib/libiomp5.so")
  set(P2O_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/paddle2onnx/lib/libpaddle2onnx.so")
  set(ORT_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onnxruntime/lib/libonnxruntime.so")
endif(WIN32)

# Path Paddle Inference ELF lib file
if(UNIX AND (NOT APPLE))
  set(PATCHELF_SCRIPT ${PROJECT_SOURCE_DIR}/scripts/patch_paddle_inference.py)
  set(PATCHELF_TARGET ${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.so)
  add_custom_target(
    patchelf_paddle_inference ALL COMMAND  bash -c 
    "PATCHELF_EXE=${PATCHELF_EXE} python  ${PATCHELF_SCRIPT} ${PATCHELF_TARGET} ${PADDLEINFERENCE_VERSION}"
    DEPENDS ${LIBRARY_NAME}
  )
  unset(PATCHELF_SCRIPT)
  unset(PATCHELF_TARGET)
endif()

add_library(external_paddle_inference STATIC IMPORTED GLOBAL)
set_property(TARGET external_paddle_inference PROPERTY IMPORTED_LOCATION
                                         ${PADDLEINFERENCE_COMPILE_LIB})
add_dependencies(external_paddle_inference ${PADDLEINFERENCE_PROJECT})


add_library(external_p2o STATIC IMPORTED GLOBAL)
set_property(TARGET external_p2o PROPERTY IMPORTED_LOCATION
        ${P2O_LIB})
add_dependencies(external_p2o ${PADDLEINFERENCE_PROJECT})

add_library(external_ort STATIC IMPORTED GLOBAL)
set_property(TARGET external_ort PROPERTY IMPORTED_LOCATION
        ${ORT_LIB})
add_dependencies(external_ort ${PADDLEINFERENCE_PROJECT})

add_library(external_dnnl STATIC IMPORTED GLOBAL)
set_property(TARGET external_dnnl PROPERTY IMPORTED_LOCATION
                                        ${DNNL_LIB})
add_dependencies(external_dnnl ${PADDLEINFERENCE_PROJECT})

add_library(external_omp STATIC IMPORTED GLOBAL)
set_property(TARGET external_omp PROPERTY IMPORTED_LOCATION
                                        ${OMP_LIB})
add_dependencies(external_omp ${PADDLEINFERENCE_PROJECT})

# Compatible policy for 2.4.x/2.5.x/2.6.x and latest dev.
if (NOT WITH_KUNLUNXIN)
  string(REGEX MATCH "0.0.0" PADDLEINFERENCE_USE_DEV ${PADDLEINFERENCE_VERSION})
  string(REGEX MATCH "2.4|post24|post2.4" PADDLEINFERENCE_USE_2_4_x ${PADDLEINFERENCE_VERSION})
  string(REGEX MATCH "2.5|post25|post2.5" PADDLEINFERENCE_USE_2_5_x ${PADDLEINFERENCE_VERSION})
  string(REGEX MATCH "2.6|post26|post2.6" PADDLEINFERENCE_USE_2_6_x ${PADDLEINFERENCE_VERSION})
endif()
  
if(PADDLEINFERENCE_USE_DEV)
  set(PADDLEINFERENCE_API_COMPAT_DEV ON CACHE BOOL "" FORCE)
endif()

if(PADDLEINFERENCE_USE_2_6_x)
  set(PADDLEINFERENCE_API_COMPAT_2_6_x ON CACHE BOOL "" FORCE)
endif()

if(PADDLEINFERENCE_USE_2_5_x)
  set(PADDLEINFERENCE_API_COMPAT_2_5_x ON CACHE BOOL "" FORCE)
endif()

if(PADDLEINFERENCE_USE_2_4_x AND (NOT PADDLEINFERENCE_API_COMPAT_2_5_x) AND (NOT PADDLEINFERENCE_API_COMPAT_2_6_x) AND (NOT PADDLEINFERENCE_API_COMPAT_DEV))
  set(PADDLEINFERENCE_API_COMPAT_2_4_x ON CACHE BOOL "" FORCE)
  message(WARNING "You are using PADDLEINFERENCE_USE_2_4_x:${PADDLEINFERENCE_VERSION}, force PADDLEINFERENCE_API_COMPAT_2_4_x=ON")
endif()

if(PADDLEINFERENCE_API_COMPAT_2_4_x)
  add_definitions(-DPADDLEINFERENCE_API_COMPAT_2_4_x)
endif()

if(PADDLEINFERENCE_API_COMPAT_2_5_x)
  add_definitions(-DPADDLEINFERENCE_API_COMPAT_2_5_x)
endif()

if(PADDLEINFERENCE_API_COMPAT_2_6_x)
  add_definitions(-DPADDLEINFERENCE_API_COMPAT_2_6_x)
endif()

if(PADDLEINFERENCE_API_COMPAT_DEV)
  add_definitions(-DPADDLEINFERENCE_API_COMPAT_DEV)
endif()

# Compatible policy for custom paddle ops
if(PADDLEINFERENCE_API_COMPAT_2_5_x AND (NOT WITH_KUNLUNXIN))
  # no c++ standard policy conflicts vs c++ 11
  # TODO: support custom ops for latest dev
  set(PADDLEINFERENCE_API_CUSTOM_OP ON CACHE BOOL "" FORCE)
  # add paddle_inference/paddle/include path for custom ops
  # the extension.h and it's deps headers are located in 
  # paddle/include/paddle directory.
  include_directories(${PADDLEINFERENCE_INC_DIR}/paddle/include)
  message(WARNING "You are using PADDLEINFERENCE_API_COMPAT_2_5_x:${PADDLEINFERENCE_VERSION}, force PADDLEINFERENCE_API_CUSTOM_OP=${PADDLEINFERENCE_API_CUSTOM_OP}")
endif()

function(set_paddle_custom_ops_compatible_policy)
  if(PADDLEINFERENCE_API_CUSTOM_OP AND (NOT WITH_KUNLUNXIN))
    if(NOT MSVC)
      # TODO: add non c++ 14 policy for latest dev
      if(NOT PADDLEINFERENCE_API_COMPAT_2_5_x)
        # gcc c++ 14 policy for 2.4.x
        if(NOT DEFINED CMAKE_CXX_STANDARD)
          set(CMAKE_CXX_STANDARD 14 PARENT_SCOPE)
          message(WARNING "Found PADDLEINFERENCE_API_CUSTOM_OP=ON, but CMAKE_CXX_STANDARD is not defined, use c++ 14 by default!")
        elseif(NOT (CMAKE_CXX_STANDARD EQUAL 14))
          set(CMAKE_CXX_STANDARD 14 PARENT_SCOPE)
          message(WARNING "Found PADDLEINFERENCE_API_CUSTOM_OP=ON, force use c++ 14!")
        endif()
      endif()
      if(WITH_GPU)
        # cuda c++ 14 policy for 2.4.x
        if(NOT PADDLEINFERENCE_API_COMPAT_2_5_x)
          if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 14 PARENT_SCOPE)
            message(WARNING "Found PADDLEINFERENCE_API_CUSTOM_OP=ON and WITH_GPU=ON, but CMAKE_CUDA_STANDARD is not defined, use c++ 14 by default!")
          elseif(NOT (CMAKE_CUDA_STANDARD EQUAL 14))
            set(CMAKE_CUDA_STANDARD 14 PARENT_SCOPE)
            message(WARNING "Found PADDLEINFERENCE_API_CUSTOM_OP=ON and WITH_GPU=ON, force use c++ 14!")
          endif()
        endif()
      endif()
    endif()
    # common compile flags for paddle custom ops
    add_definitions(-DPADDLE_ON_INFERENCE)
    add_definitions(-DPADDLE_NO_PYTHON)
    if(WITH_GPU)
      add_definitions(-DPADDLE_WITH_CUDA)
    endif()
  endif()
endfunction()
