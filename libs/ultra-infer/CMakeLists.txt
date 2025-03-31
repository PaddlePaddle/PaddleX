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

PROJECT(ultra_infer C CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)


option(CSRCS_DIR_NAME "Name of source code directory")
option(LIBRARY_NAME "Name of build library name")
option(PY_LIBRARY_NAME "Name of build python library name")
if(NOT CSRCS_DIR_NAME)
  set(CSRCS_DIR_NAME ".")
endif()
if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME "ultra_infer")
endif()
if(NOT PY_LIBRARY_NAME)
  set(PY_LIBRARY_NAME "ultra_infer_main")
endif()

include(ExternalProject)
set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third_party)

add_subdirectory(${CSRCS_DIR_NAME}/ultra_infer)
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
  endif()
  set(CMAKE_CXX_FLAGS "-Wno-format -g0 -O3")
  if(NEED_ABI0)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  endif()
endif(NOT MSVC)

include(${PROJECT_SOURCE_DIR}/cmake/build_tools.cmake)
if(UNIX AND (NOT APPLE) AND (NOT WITH_TIMVX))
  download_patchelf()
  set(PATCHELF_EXE ${THIRD_PARTY_PATH}/patchelf/bin/patchelf)
endif()


############################# Basic Options for UltraInfer ################################
option(WITH_GPU "Whether WITH_GPU=ON, will enable onnxruntime-gpu/paddle-infernce-gpu/poros-gpu" OFF)
option(WITH_IPU "Whether WITH_IPU=ON, will enable paddle-infernce-ipu" OFF)
option(WITH_OPENCL "Whether WITH_OPENCL=ON, will enable paddle-lite-gpu" OFF)
option(ENABLE_ORT_BACKEND "Whether to enable onnxruntime backend." OFF)
option(ENABLE_TRT_BACKEND "Whether to enable tensorrt backend." OFF)
option(ENABLE_PADDLE_BACKEND "Whether to enable paddle backend." OFF)
option(ENABLE_POROS_BACKEND "Whether to enable poros backend." OFF)
option(ENABLE_OPENVINO_BACKEND "Whether to enable openvino backend." OFF)
option(ENABLE_RKNPU2_BACKEND "Whether to enable RKNPU2 backend." OFF)
option(ENABLE_SOPHGO_BACKEND "Whether to enable SOPHON backend." OFF)
option(ENABLE_TVM_BACKEND "Whether to enable TVM backend." OFF)
option(ENABLE_LITE_BACKEND "Whether to enable paddle lite backend." OFF)
option(ENABLE_HORIZON_BACKEND "Whether to enable HORIZON backend." OFF)
option(ENABLE_OM_BACKEND "Whether to enable OM backend." OFF)
option(ENABLE_VISION "Whether to enable vision models usage." OFF)
option(ENABLE_TEXT "Whether to enable text models usage." OFF)
option(ENABLE_FLYCV "Whether to enable flycv to boost image preprocess." OFF)
option(ENABLE_CVCUDA "Whether to enable NVIDIA CV-CUDA to boost image preprocess." OFF)
option(ENABLE_BENCHMARK "Whether to enable Benchmark mode." OFF)
option(WITH_ASCEND "Whether to compile for Huawei Ascend deploy." OFF)
option(WITH_DIRECTML "Whether to compile for onnxruntime DirectML deploy." OFF)
option(WITH_TIMVX "Whether to compile for TIMVX deploy." OFF)
option(WITH_KUNLUNXIN "Whether to compile for KunlunXin XPU deploy." OFF)
option(WITH_TESTING "Whether to compile with unittest." OFF)
option(WITH_CAPI "Whether to compile with c api." OFF)
option(WITH_CSHARPAPI "Whether to compile with c# api" OFF)

option(BUILD_EXAMPLES "Whether to build ultra_infer with vision examples" OFF)
option(BUILD_PADDLE2ONNX "Whether to build paddle2onnx from sources" OFF)

option(BUILD_FD_TRITON_BACKEND "Whether to compile as Triton Inference Server backend." OFF)

######################### Paths to user's custom libraries directory #####################
set(CUDA_DIRECTORY "" CACHE PATH "If build tensorrt backend, need to define path of cuda library.")
set(TRT_DIRECTORY "" CACHE PATH "If build tensorrt backend, need to define path of tensorrt library.")
set(ORT_DIRECTORY "" CACHE PATH "User can specify the installed onnxruntime directory.")
set(OPENCV_DIRECTORY "" CACHE PATH "User can specify the installed opencv directory.")
set(OPENVINO_DIRECTORY "" CACHE PATH "User can specify the installed openvino directory.")

# Whether to build ultra_infer on device Nvidia Jetson
# Only support CPU Inference & GPU(TensorRT) Inference Now
option(BUILD_ON_JETSON "Whether to build ultra_infer on Nvidia Jetson" OFF)
if(BUILD_ON_JETSON)
  set(WITH_GPU ON)
  set(ENABLE_TRT_BACKEND ON)
  set(ENABLE_ORT_BACKEND ON)
endif()

# config GIT_URL with github mirrors to speed up dependent repos clone
option(GIT_URL "Git URL to clone dependent repos" ${GIT_URL})
if(NOT GIT_URL)
    set(GIT_URL "https://github.com")
endif()

# check build options
include(${PROJECT_SOURCE_DIR}/cmake/check.cmake)

if(WIN32)
  add_definitions(-DYAML_CPP_DLL)
  set(YAML_BUILD_SHARED_LIBS ON)
  set(YAML_CPP_INSTALL ON)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif()

if(NOT CUDA_DIRECTORY)
  set(CUDA_DIRECTORY "/usr/local/cuda")
endif()

option(BUILD_ULTRAINFER_PYTHON "if build python lib for ultra_infer." OFF)

set(HEAD_DIR "${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}")
include_directories(${HEAD_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (WITH_TIMVX)
  include(${PROJECT_SOURCE_DIR}/cmake/timvx.cmake)
endif()

if (WITH_ASCEND)
  include(${PROJECT_SOURCE_DIR}/cmake/ascend.cmake)
endif()

if (WITH_KUNLUNXIN)
  include(${PROJECT_SOURCE_DIR}/cmake/kunlunxin.cmake)
endif()

if(WITH_IPU)
  if(NOT ENABLE_PADDLE_BACKEND)
    message("Will force to set ENABLE_PADDLE_BACKEND when build with GraphCore IPU.")
    set(ENABLE_PADDLE_BACKEND ON)
  endif()
  add_definitions(-DWITH_IPU)
endif()

# Check for macOS architecture
get_osx_architecture()

##################################### Building: UltraInfer C++ SDK #######################################
add_definitions(-DULTRAINFER_LIB)
# set CMAKE_BUILD_TYPE to Release
add_definitions(-DCMAKE_BUILD_TYPE=Release)
# configure files before glob sources.
configure_file(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/core/config.h.in ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/core/config.h)
configure_file(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/pybind/main.cc.in ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/pybind/main.cc)
file(GLOB_RECURSE ALL_DEPLOY_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/*.cc)
file(GLOB_RECURSE DEPLOY_ORT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/ort/*.cc)
file(GLOB_RECURSE DEPLOY_PADDLE_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/paddle/*.cc)
file(GLOB_RECURSE DEPLOY_POROS_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/poros/*.cc)
file(GLOB_RECURSE DEPLOY_TRT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/tensorrt/*.cc ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/tensorrt/*.cpp)
file(GLOB_RECURSE DEPLOY_OPENVINO_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/openvino/*.cc)
file(GLOB_RECURSE DEPLOY_RKNPU2_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/rknpu2/*.cc)
file(GLOB_RECURSE DEPLOY_HORIZON_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/horizon/*.cc)
file(GLOB_RECURSE DEPLOY_SOPHGO_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/sophgo/*.cc)
file(GLOB_RECURSE DEPLOY_TVM_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/tvm/*.cc)
file(GLOB_RECURSE DEPLOY_LITE_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/lite/*.cc)
file(GLOB_RECURSE DEPLOY_PIPELINE_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/pipeline/*.cc)
file(GLOB_RECURSE DEPLOY_OM_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/om/*.cc)
file(GLOB_RECURSE DEPLOY_VISION_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/vision/*.cc)
file(GLOB_RECURSE DEPLOY_TEXT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/text/*.cc)
file(GLOB_RECURSE DEPLOY_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/pybind/*.cc ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/*_pybind.cc)
file(GLOB_RECURSE DEPLOY_PADDLE_CUSTOM_OP_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/paddle/ops/*.cc)
if(WITH_GPU)
  file(GLOB_RECURSE DEPLOY_CUDA_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/*.cu)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_CUDA_SRCS})
  file(GLOB_RECURSE DEPLOY_PADDLE_CUSTOM_OP_CUDA_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/paddle/ops/*.cu)
  list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_PADDLE_CUSTOM_OP_CUDA_SRCS})
  file(GLOB_RECURSE DEPLOY_VISION_CUDA_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/vision/*.cu)
  list(APPEND DEPLOY_VISION_SRCS ${DEPLOY_VISION_CUDA_SRCS})
  file(GLOB_RECURSE DEPLOY_TEXT_CUDA_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/text/*.cu)
  list(APPEND DEPLOY_TEXT_SRCS ${DEPLOY_TEXT_CUDA_SRCS})
endif()
list(REMOVE_ITEM DEPLOY_PADDLE_SRCS ${DEPLOY_PADDLE_CUSTOM_OP_SRCS})
list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_ORT_SRCS} ${DEPLOY_PADDLE_SRCS}
                                 ${DEPLOY_POROS_SRCS} ${DEPLOY_TRT_SRCS}
                                 ${DEPLOY_OPENVINO_SRCS} ${DEPLOY_LITE_SRCS}
                                 ${DEPLOY_VISION_SRCS} ${DEPLOY_TEXT_SRCS}
                                 ${DEPLOY_PIPELINE_SRCS} ${DEPLOY_RKNPU2_SRCS}
                                 ${DEPLOY_SOPHGO_SRCS}
                                 ${DEPLOY_HORIZON_SRCS} ${DEPLOY_TVM_SRCS}
                                 ${DEPLOY_PADDLE_CUSTOM_OP_SRCS} ${DEPLOY_OM_SRCS})


set(DEPEND_LIBS "")

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" ULTRAINFER_VERSION)
string(STRIP "${ULTRAINFER_VERSION}" ULTRAINFER_VERSION)

# Add eigen lib
download_eigen()
include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
if(WIN32)
  add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

# sw(sunway) not support thread_local semantic
if(WITH_SW)
  add_definitions(-DEIGEN_AVOID_THREAD_LOCAL)
endif()

if(ENABLE_ORT_BACKEND)
  set(ENABLE_PADDLE2ONNX ON)
  add_definitions(-DENABLE_ORT_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_ORT_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/onnxruntime.cmake)
  list(APPEND DEPEND_LIBS external_onnxruntime)
endif()

if(ENABLE_LITE_BACKEND)
  add_definitions(-DENABLE_LITE_BACKEND)
  include(${PROJECT_SOURCE_DIR}/cmake/paddlelite.cmake)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_LITE_SRCS})
  list(APPEND DEPEND_LIBS external_paddle_lite)
endif()

if(ENABLE_PADDLE_BACKEND)
  set(ENABLE_PADDLE2ONNX ON)
  add_definitions(-DENABLE_PADDLE_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_PADDLE_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/paddle_inference.cmake)
  list(APPEND DEPEND_LIBS external_paddle_inference)
  if(external_dnnl_FOUND)
    list(APPEND DEPEND_LIBS external_dnnl external_omp)
  endif()
  if(external_ort_FOUND)
    list(APPEND DEPEND_LIBS external_p2o external_ort)
  endif()
  if(PADDLEINFERENCE_API_CUSTOM_OP)
    set_paddle_custom_ops_compatible_policy()
    list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_PADDLE_CUSTOM_OP_SRCS})
    if(WITH_GPU)
      list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_PADDLE_CUSTOM_OP_CUDA_SRCS})
    endif()
  endif()
endif()

if(ENABLE_OPENVINO_BACKEND)
  set(ENABLE_PADDLE2ONNX ON)
  add_definitions(-DENABLE_OPENVINO_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_OPENVINO_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/openvino.cmake)
endif()

if(ENABLE_RKNPU2_BACKEND)
  add_definitions(-DENABLE_RKNPU2_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_RKNPU2_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/rknpu2.cmake)
  list(APPEND DEPEND_LIBS ${RKNN_RT_LIB})
endif()

if(ENABLE_HORIZON_BACKEND)
  add_definitions(-DENABLE_HORIZON_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_HORIZON_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/horizon.cmake)
  list(APPEND DEPEND_LIBS ${BPU_libs})
endif()

if(ENABLE_TVM_BACKEND)
  set(CMAKE_CXX_STANDARD 17)
  add_definitions(-DENABLE_TVM_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TVM_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/tvm.cmake)
  list(APPEND DEPEND_LIBS ${TVM_RUNTIME_LIB})
endif()

if(ENABLE_SOPHGO_BACKEND)
  add_definitions(-DENABLE_SOPHGO_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_SOPHGO_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/sophgo.cmake)
  list(APPEND DEPEND_LIBS ${SOPHGO_RT_LIB})
endif()

if(ENABLE_POROS_BACKEND)
  set(CMAKE_CXX_STANDARD 14)
  add_definitions(-DENABLE_POROS_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_POROS_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/poros.cmake)
  list(APPEND DEPEND_LIBS external_poros)
  set(PYTHON_MINIMUM_VERSION 3.6)
  set(PYTORCH_MINIMUM_VERSION 1.9)
  set(TENSORRT_MINIMUM_VERSION 8.0)
  # find python3
  find_package(Python3 ${PYTHON_MINIMUM_VERSION} REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Found Python: ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")

  if (NOT Python3_SITELIB)
    message(FATAL_ERROR "site-packages not found. ")
  else ()
    message(STATUS "site-packages: ${Python3_SITELIB}")
  endif ()
  include_directories(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/poros/common)
  # find trt
  if(NOT WITH_GPU)
  message(FATAL_ERROR "While -DENABLE_POROS_BACKEND=ON, must set -DWITH_GPU=ON, but now it's OFF")
  endif()
  if(NOT TRT_DIRECTORY)
    message(FATAL_ERROR "While -DENABLE_POROS_BACKEND=ON, must define -DTRT_DIRECTORY, e.g -DTRT_DIRECTORY=/Downloads/TensorRT-8.4")
  endif()
  include_directories(${TRT_DIRECTORY}/include)
  find_library(TRT_INFER_LIB nvinfer ${TRT_DIRECTORY}/lib)
  find_library(TRT_ONNX_LIB nvonnxparser ${TRT_DIRECTORY}/lib)
  find_library(TRT_PLUGIN_LIB nvinfer_plugin ${TRT_DIRECTORY}/lib)
  list(APPEND DEPEND_LIBS ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_PLUGIN_LIB})
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
  endif()
  if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
  endif()
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  message(STATUS "Copying ${TRT_DIRECTORY}/lib to ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib ...")
  execute_process(COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/copy_directory.py ${TRT_DIRECTORY}/lib ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib)
endif()

if(ENABLE_OM_BACKEND)
  add_definitions(-DENABLE_OM_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_OM_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/om.cmake)
  list(APPEND DEPEND_LIBS ${NPU_libs})
endif()

if(WITH_GPU)
  add_definitions(-DWITH_GPU)
  include_directories(${CUDA_DIRECTORY}/include)
  if(WIN32)
    find_library(CUDA_LIB cudart ${CUDA_DIRECTORY}/lib/x64)
    find_library(NVJPEG_LIB nvjpeg ${CUDA_DIRECTORY}/lib/x64)
    add_definitions(-DENABLE_NVJPEG)
  else()
    find_library(CUDA_LIB cudart ${CUDA_DIRECTORY}/lib64)
    if(NOT BUILD_ON_JETSON)
      find_library(NVJPEG_LIB nvjpeg ${CUDA_DIRECTORY}/lib64)
      add_definitions(-DENABLE_NVJPEG)
    endif()
  endif()
  list(APPEND DEPEND_LIBS ${CUDA_LIB} ${NVJPEG_LIB})

  # build CUDA source files in ultra_infer, CUDA source files include CUDA preprocessing, TRT plugins, etc.
  enable_language(CUDA)
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}, version: "
                  "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
  include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)
endif()

if(WITH_OPENCL)
  add_definitions(-DWITH_OPENCL)
endif()

if(ENABLE_TRT_BACKEND)
  set(ENABLE_PADDLE2ONNX ON)
  if(APPLE OR IOS)
    message(FATAL_ERROR "Cannot enable tensorrt backend in mac/ios os, please set -DENABLE_TRT_BACKEND=OFF.")
  endif()
  if(NOT WITH_GPU)
    message(FATAL_ERROR "While -DENABLE_TRT_BACKEND=ON, must set -DWITH_GPU=ON, but now it's OFF")
  endif()
  if(NOT BUILD_ON_JETSON)
    if(NOT TRT_DIRECTORY)
      set(TRT_INC_DIR /usr/include/x86_64-linux-gnu/)
      set(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu/)
    endif()
  endif()
  if(BUILD_ON_JETSON)
    set(TRT_INC_DIR /usr/include/aarch64-linux-gnu/)
    set(TRT_LIB_DIR /usr/lib/aarch64-linux-gnu/)
  else()
    set(TRT_INC_DIR /usr/include/x86_64-linux-gnu/)
    set(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu/)
    if(TRT_DIRECTORY)
      set(TRT_INC_DIR ${TRT_DIRECTORY}/include)
      set(TRT_LIB_DIR ${TRT_DIRECTORY}/lib)
    endif()
  endif()

  add_definitions(-DENABLE_TRT_BACKEND)
  include_directories(${TRT_INC_DIR})
  include_directories(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/tensorrt/common)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TRT_SRCS})
  find_library(TRT_INFER_LIB nvinfer ${TRT_LIB_DIR} NO_DEFAULT_PATH)
  find_library(TRT_ONNX_LIB nvonnxparser ${TRT_LIB_DIR} NO_DEFAULT_PATH)
  find_library(TRT_PLUGIN_LIB nvinfer_plugin ${TRT_LIB_DIR} NO_DEFAULT_PATH)
  list(APPEND DEPEND_LIBS ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_PLUGIN_LIB})

  if(NOT BUILD_ON_JETSON AND TRT_DIRECTORY)
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
    endif()
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
      file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
    endif()

    if (NOT Python_EXECUTABLE)
      find_package(Python COMPONENTS Interpreter Development REQUIRED)
    endif()

    message(STATUS "Copying ${TRT_DIRECTORY}/lib to ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib ...")
    execute_process(COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/copy_directory.py ${TRT_DIRECTORY}/lib ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib)
    file(GLOB_RECURSE TRT_STATIC_LIBS ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib/*.a)
    if(TRT_STATIC_LIBS)
      file(REMOVE ${TRT_STATIC_LIBS})
    endif()
    if(UNIX AND (NOT APPLE))
      execute_process(COMMAND sh -c "ls *.so*" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib
	      COMMAND sh -c "xargs ${PATCHELF_EXE} --force-rpath --set-rpath '$ORIGIN'" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib
    	          RESULT_VARIABLE result
                      OUTPUT_VARIABLE curr_out
                      ERROR_VARIABLE  curr_out)
      if(ret EQUAL "1")
	     message(FATAL_ERROR "Failed to patchelf tensorrt libraries.")
      endif()
      message(STATUS "result:${result} out:${curr_out}")
    endif()
  endif()
endif()

if(ENABLE_VISION)
  add_definitions(-DENABLE_VISION)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_VISION_SRCS})
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_PIPELINE_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/opencv.cmake)

  if(ENABLE_FLYCV)
    add_definitions(-DENABLE_FLYCV)
    include(${PROJECT_SOURCE_DIR}/cmake/flycv.cmake)
    list(APPEND DEPEND_LIBS ${FLYCV_LIBRARIES})
  endif()

  if(ENABLE_CVCUDA)
    include(${PROJECT_SOURCE_DIR}/cmake/cvcuda.cmake)
    add_definitions(-DENABLE_CVCUDA)
    list(APPEND DEPEND_LIBS nvcv_types cvcuda)
  endif()
endif()

download_yaml_cpp()
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp)
list(APPEND DEPEND_LIBS yaml-cpp)
include_directories(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include)

if(ENABLE_TEXT)
  add_definitions(-DENABLE_TEXT)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TEXT_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/fast_tokenizer.cmake)
endif()

if(ENABLE_PADDLE2ONNX)
  add_definitions(-DENABLE_PADDLE2ONNX)
  if(BUILD_PADDLE2ONNX)
    download_protobuf()
    download_onnx()
    download_optimizer()
    include(${PROJECT_SOURCE_DIR}/cmake/build_paddle2onnx.cmake)
    list(APPEND ALL_DEPLOY_SRCS ${PADDLE2ONNX_ALL_SRCS})
    list(APPEND DEPEND_LIBS p2o_paddle_proto onnx)
  else()
    include(${PROJECT_SOURCE_DIR}/cmake/paddle2onnx.cmake)
    list(APPEND DEPEND_LIBS external_paddle2onnx)
  endif()
endif(ENABLE_PADDLE2ONNX)

if(WITH_CAPI)
  include(${PROJECT_SOURCE_DIR}/c_api/CMakeLists.txt)
  if(MSVC)
  add_definitions(-DFD_CAPI)
  endif()
endif()

if(WITH_CSHARPAPI)
  if(MSVC)
  add_subdirectory(${PROJECT_SOURCE_DIR}/csharp)
  endif()
endif()

configure_file(${PROJECT_SOURCE_DIR}/UltraInfer.cmake.in ${PROJECT_SOURCE_DIR}/UltraInfer.cmake @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/UltraInferCSharp.cmake.in ${PROJECT_SOURCE_DIR}/UltraInferCSharp.cmake @ONLY)
if(BUILD_FD_TRITON_BACKEND)
  configure_file(${PROJECT_SOURCE_DIR}/python/ultra_infer/c_lib_wrap.py.in ${PROJECT_SOURCE_DIR}/python/ultra_infer/c_lib_wrap.py)
else()
  configure_file(${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/c_lib_wrap.py.in ${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/c_lib_wrap.py)
endif()
configure_file(${PROJECT_SOURCE_DIR}/python/scripts/process_libraries.py.in ${PROJECT_SOURCE_DIR}/python/scripts/process_libraries.py)

list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_PYBIND_SRCS})

add_library(${LIBRARY_NAME} SHARED ${ALL_DEPLOY_SRCS})

redefine_file_macro(${LIBRARY_NAME})

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" ULTRAINFER_VERSION)
string(STRIP "${ULTRAINFER_VERSION}" ULTRAINFER_VERSION)
if (APPLE)
  set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
elseif(MSVC)
else()
  if(WITH_GPU)
    set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${LIBRARY_NAME} PROPERTIES INTERFACE_COMPILE_OPTIONS
       "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-fvisibility=hidden>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fvisibility=hidden>")
  else()
    set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  endif()
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${ULTRAINFER_VERSION})
if(MSVC)
  # disable warnings for dll export
  target_compile_options(${LIBRARY_NAME} PRIVATE "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:/wd4251>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/wd4251>")
  file(GLOB FD_FILES_REQUIRE_BIGOBJ ${CSRCS_DIR_NAME}/ultra_infer/function/reduce.cc)
  set_source_files_properties(${FD_FILES_REQUIRE_BIGOBJ} PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

target_link_libraries(${LIBRARY_NAME} ${DEPEND_LIBS})

##################################### Examples ####################################
if(WIN32)
  if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    add_custom_target(copy_yaml_library ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/third_party/yaml-cpp  ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/yaml-cpp/lib DEPENDS ${LIBRARY_NAME})
  else()
    add_custom_target(copy_yaml_library ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/third_party/yaml-cpp/Release  ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/yaml-cpp/lib DEPENDS ${LIBRARY_NAME})
    add_custom_target(copy_yaml_include ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include  ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/yaml-cpp/include DEPENDS ${LIBRARY_NAME})
  endif()
endif()

# add examples after prepare include paths for third-parties
if(BUILD_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  add_definitions(-DBUILD_EXAMPLES)
  if(NOT EXECUTABLE_OUTPUT_PATH STREQUAL ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)
  endif()
  include(${PROJECT_SOURCE_DIR}/cmake/gflags.cmake)
  add_subdirectory(examples)
endif()

if (WITH_TESTING AND EXISTS ${PROJECT_SOURCE_DIR}/tests)
  add_definitions(-DWITH_TESTING)
  include(${PROJECT_SOURCE_DIR}/cmake/gtest.cmake)
  if(NOT BUILD_EXAMPLES)
    include(${PROJECT_SOURCE_DIR}/cmake/gflags.cmake)
  endif()
  include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
  add_subdirectory(tests)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/summary.cmake)
ultra_infer_summary()

################################ Installation: UltraInfer C++ SDK ###############################
if(WIN32)
  install(
    TARGETS ${LIBRARY_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib
  )
else()
  install(
    TARGETS ${LIBRARY_NAME}
    LIBRARY DESTINATION lib)
endif()

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/runtime/backends/*/*.h"
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install
  DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs
)

install(
  FILES
  ${PROJECT_SOURCE_DIR}/LICENSE
  ${PROJECT_SOURCE_DIR}/ThirdPartyNotices.txt
  ${PROJECT_SOURCE_DIR}/VERSION_NUMBER
  ${PROJECT_SOURCE_DIR}/UltraInfer.cmake
  ${PROJECT_SOURCE_DIR}/UltraInferCSharp.cmake
  ${PROJECT_SOURCE_DIR}/cmake/UltraInferConfig.cmake
  ${PROJECT_SOURCE_DIR}/cmake/utils.cmake
  ${PROJECT_SOURCE_DIR}/cmake/summary.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(
  FILES ${PROJECT_SOURCE_DIR}/cmake/gflags.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/utils
)

if(NOT WIN32)
  install(
    FILES ${PROJECT_SOURCE_DIR}/scripts/ultra_infer_init.sh
    DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
else()
  install(
    FILES ${PROJECT_SOURCE_DIR}/scripts/ultra_infer_init.bat
    DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
endif()

if(WITH_ASCEND)
  install(
    FILES ${PROJECT_SOURCE_DIR}/scripts/ascend_init.sh
    DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
endif()

if(WITH_CAPI)
  install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/c_api/ultra_infer_capi
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*/types_internal.h" EXCLUDE
  )
endif()

include(${PROJECT_SOURCE_DIR}/cmake/config_cpack.cmake)

if(WIN32 AND BUILD_EXAMPLES)
  get_windows_path(_tmp_install_dir ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install)
  get_windows_path(_publish_exe_dir ${EXECUTABLE_OUTPUT_PATH}/Release)
  list(GET CMAKE_CONFIGURATION_TYPES 0 _CONFIG_TYPE)
  if((${CMAKE_BUILD_TYPE} MATCHES "Release") OR (${_CONFIG_TYPE} MATCHES "Release"))
    install(TARGETS ${LIBRARY_NAME} RUNTIME DESTINATION ${EXECUTABLE_OUTPUT_PATH}/Release)
    add_custom_target(
      copy_fd_third_dlls_examples ALL COMMAND
      cmd /C ${PROJECT_SOURCE_DIR}/scripts/ultra_infer_init.bat install ${_tmp_install_dir} ${_publish_exe_dir} noconfirm)
    add_dependencies(copy_fd_third_dlls_examples ${LIBRARY_NAME} copy_yaml_library)
  endif()
endif()

############################### Building: UltraInfer Python Wheel #############################
if(BUILD_ULTRAINFER_PYTHON)
  add_definitions(-DBUILD_ULTRAINFER_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # find_package Python has replaced PythonInterp and PythonLibs since cmake 3.12
  # Use the following command in the future; now this is only compatible with the latest pybind11
  # find_package(Python ${PY_VERSION} COMPONENTS Interpreter Development REQUIRED)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION})
  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
  endif()

  if(NOT ENABLE_VISION)
    file(GLOB_RECURSE VISION_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/vision/*_pybind.cc)
    file(GLOB_RECURSE PIPELINE_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/pipeline/*_pybind.cc)
    list(REMOVE_ITEM DEPLOY_PYBIND_SRCS ${VISION_PYBIND_SRCS} ${PIPELINE_PYBIND_SRCS})
  endif()

  if (NOT ENABLE_TEXT)
    file(GLOB_RECURSE TEXT_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/ultra_infer/text/*_pybind.cc)
    list(REMOVE_ITEM DEPLOY_PYBIND_SRCS ${TEXT_PYBIND_SRCS})
  endif()

  add_library(${PY_LIBRARY_NAME} MODULE ${DEPLOY_PYBIND_SRCS})
  redefine_file_macro(${PY_LIBRARY_NAME})
  set_target_properties(${PY_LIBRARY_NAME} PROPERTIES PREFIX "")
  set_target_properties(${PY_LIBRARY_NAME}
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(${PY_LIBRARY_NAME} PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(${PY_LIBRARY_NAME}
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(${PY_LIBRARY_NAME} PRIVATE
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                             $<INSTALL_INTERFACE:include>
                             ${PYTHON_INCLUDE_DIR})

  download_pybind()
  target_include_directories(${PY_LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
  download_dlpack()
  target_include_directories(${PY_LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/dlpack/include)

  if(APPLE)
    set_target_properties(${PY_LIBRARY_NAME}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  target_link_libraries(${PY_LIBRARY_NAME} PUBLIC ${LIBRARY_NAME})

  if(MSVC)
    target_link_libraries(${PY_LIBRARY_NAME} PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(${PY_LIBRARY_NAME}
                           PRIVATE /MP
                                   /wd4244 # 'argument': conversion from 'google::
                                           # protobuf::uint64' to 'int', possible
                                           # loss of data
                                   /wd4267 # Conversion from 'size_t' to 'int',
                                           # possible loss of data
                                   /wd4996 # The second parameter is ignored.
                                   ${EXTRA_FLAGS})
    target_compile_options(${PY_LIBRARY_NAME} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MT> $<$<CONFIG:Debug>:/MTd>)
  endif()

  file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/libs)
  file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/libs)

  if(WIN32)
    add_custom_target(copy_fd_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy_directory   ${CMAKE_CURRENT_BINARY_DIR}/Release ${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/libs/ DEPENDS ${PY_LIBRARY_NAME})
  elseif(APPLE)
    add_custom_target(copy_fd_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/*.so** ${CMAKE_CURRENT_BINARY_DIR}/*.dylib** ${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/libs/ DEPENDS ${PY_LIBRARY_NAME})
  else()
    add_custom_target(copy_fd_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/*.so* ${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/libs/ DEPENDS ${PY_LIBRARY_NAME})
  endif()
  add_custom_target(copy_third_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install ${PROJECT_SOURCE_DIR}/python/${LIBRARY_NAME}/libs/third_libs DEPENDS ${PY_LIBRARY_NAME})
endif(BUILD_ULTRAINFER_PYTHON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4.0")
    string(STRIP "${CMAKE_CXX_COMPILER_VERSION}" CMAKE_CXX_COMPILER_VERSION)
    message(FATAL_ERROR "[ERROR] UltraInfer require g++ version >= 5.4.0, but now your g++ version is ${CMAKE_CXX_COMPILER_VERSION}, this may cause failure! Use -DCMAKE_CXX_COMPILER to define path of your compiler.")
  endif()
endif()
