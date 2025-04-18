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

if(NOT ENABLE_TRT_BACKEND)
  message(FATAL_ERROR "While ENABLE_POROS_BACKEND, requires ENABLE_TRT_BACKEND=ON, but now its OFF.")
endif()

set(POROS_PROJECT "extern_poros")
set(POROS_PREFIX_DIR ${THIRD_PARTY_PATH}/poros)
set(POROS_SOURCE_DIR
    ${THIRD_PARTY_PATH}/poros/src/${POROS_PROJECT})
set(POROS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/poros)
set(POROS_INC_DIR
    "${POROS_INSTALL_DIR}/include"
    CACHE PATH "poros include directory." FORCE)
set(POROS_LIB_DIR
    "${POROS_INSTALL_DIR}/lib/"
    CACHE PATH "poros lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${POROS_LIB_DIR}")

include_directories(${POROS_INC_DIR})
if(WIN32)
  message(FATAL_ERROR "Poros Backend doesn't support Windows now.")
elseif(APPLE)
  message(FATAL_ERROR "Poros Backend doesn't support Mac OSX now.")
else()
  set(POROS_COMPILE_LIB
      "${POROS_INSTALL_DIR}/lib/libporos.so"
      CACHE FILEPATH "poros compile library." FORCE)
endif(WIN32)

set(POROS_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
set(POROS_VERSION "0.1.0")
if(WIN32)
  message(FATAL_ERROR "Poros Backend doesn't support Windows now.")
elseif(APPLE)
  message(FATAL_ERROR "Poros Backend doesn't support Mac OSX now.")
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(FATAL_ERROR "Poros Backend doesn't support linux aarch64 now.")
  else()
    if(WITH_GPU)
        set(POROS_FILE "poros_manylinux_torch1.12.1_cu116_trt8.4_gcc82-${POROS_VERSION}.tar.gz")
    else()
      message(FATAL_ERROR "Poros currently only provides precompiled packages for the GPU version.")
    endif()
  endif()
endif()
set(POROS_URL "${POROS_URL_BASE}${POROS_FILE}")

ExternalProject_Add(
  ${POROS_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${POROS_URL}
  PREFIX ${POROS_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${POROS_SOURCE_DIR} ${POROS_INSTALL_DIR}
  BUILD_BYPRODUCTS ${POROS_COMPILE_LIB})

add_library(external_poros STATIC IMPORTED GLOBAL)
set_property(TARGET external_poros PROPERTY IMPORTED_LOCATION
                                         ${POROS_COMPILE_LIB})
add_dependencies(external_poros ${POROS_PROJECT})

# Download libtorch.so with ABI=1
set(TORCH_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
set(TORCH_FILE "libtorch-cxx11-abi-shared-with-deps-1.12.1-cu116.zip")
set(TORCH_URL "${TORCH_URL_BASE}${TORCH_FILE}")
message(STATUS "Use the default Torch lib from: ${TORCH_URL}")
download_and_decompress(${TORCH_URL} ${CMAKE_CURRENT_BINARY_DIR}/${TORCH_FILE} ${THIRD_PARTY_PATH}/install)
if(EXISTS ${THIRD_PARTY_PATH}/install/torch)
  file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/torch) 
endif()
file(RENAME ${THIRD_PARTY_PATH}/install/libtorch/ ${THIRD_PARTY_PATH}/install/torch)
set(TORCH_INCLUDE_DIRS ${THIRD_PARTY_PATH}/install/torch/include)
find_library(TORCH_LIBRARY torch ${THIRD_PARTY_PATH}/install/torch/lib NO_DEFAULT_PATH)
include_directories(${TORCH_INCLUDE_DIRS})
list(APPEND DEPEND_LIBS ${TORCH_LIBRARY})
