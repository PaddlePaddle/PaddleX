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

set(PADDLE2ONNX_PROJECT "extern_paddle2onnx")
set(PADDLE2ONNX_PREFIX_DIR ${THIRD_PARTY_PATH}/paddle2onnx)
set(PADDLE2ONNX_SOURCE_DIR
    ${THIRD_PARTY_PATH}/paddle2onnx/src/${PADDLE2ONNX_PROJECT})
set(PADDLE2ONNX_INSTALL_DIR ${THIRD_PARTY_PATH}/install/paddle2onnx)
set(PADDLE2ONNX_INC_DIR
    "${PADDLE2ONNX_INSTALL_DIR}/include"
    CACHE PATH "paddle2onnx include directory." FORCE)
set(PADDLE2ONNX_LIB_DIR
    "${PADDLE2ONNX_INSTALL_DIR}/lib/"
    CACHE PATH "paddle2onnx lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${PADDLE2ONNX_LIB_DIR}")

include_directories(BEFORE ${PADDLE2ONNX_INC_DIR})
if(WIN32)
  set(PADDLE2ONNX_COMPILE_LIB
      "${PADDLE2ONNX_INSTALL_DIR}/lib/paddle2onnx.lib"
      CACHE FILEPATH "paddle2onnx compile library." FORCE)
elseif(APPLE)
  set(PADDLE2ONNX_COMPILE_LIB
      "${PADDLE2ONNX_INSTALL_DIR}/lib/libpaddle2onnx.dylib"
      CACHE FILEPATH "paddle2onnx compile library." FORCE)
else()
  set(PADDLE2ONNX_COMPILE_LIB
      "${PADDLE2ONNX_INSTALL_DIR}/lib/libpaddle2onnx.so"
      CACHE FILEPATH "paddle2onnx compile library." FORCE)
endif(WIN32)

if (NOT PADDLE2ONNX_URL)
  # Use default paddle2onnx url if custom url is not setting
  set(PADDLE2ONNX_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
  set(PADDLE2ONNX_VERSION "1.0.8rc")
  if(WIN32)
    set(PADDLE2ONNX_FILE "paddle2onnx-win-x64-${PADDLE2ONNX_VERSION}.zip")
    if(NOT CMAKE_CL_64)
      set(PADDLE2ONNX_FILE "paddle2onnx-win-x86-${PADDLE2ONNX_VERSION}.zip")
    endif()
  elseif(APPLE)
    if(CURRENT_OSX_ARCH MATCHES "arm64")
      set(PADDLE2ONNX_FILE "paddle2onnx-osx-arm64-${PADDLE2ONNX_VERSION}.tgz")
    else()
      set(PADDLE2ONNX_FILE "paddle2onnx-osx-x86_64-${PADDLE2ONNX_VERSION}.tgz")
    endif()
  else()
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(PADDLE2ONNX_FILE "paddle2onnx-linux-aarch64-${PADDLE2ONNX_VERSION}.tgz")
    else()
      set(PADDLE2ONNX_FILE "paddle2onnx-linux-x64-${PADDLE2ONNX_VERSION}.tgz")
    endif()
  endif()
  set(PADDLE2ONNX_URL "${PADDLE2ONNX_URL_BASE}${PADDLE2ONNX_FILE}")
endif()

ExternalProject_Add(
  ${PADDLE2ONNX_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${PADDLE2ONNX_URL}
  PREFIX ${PADDLE2ONNX_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E remove_directory ${PADDLE2ONNX_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E make_directory ${PADDLE2ONNX_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E rename ${PADDLE2ONNX_SOURCE_DIR}/lib/
    ${PADDLE2ONNX_LIB_DIR} && ${CMAKE_COMMAND} -E copy_directory
    ${PADDLE2ONNX_SOURCE_DIR}/include ${PADDLE2ONNX_INC_DIR}
  BUILD_BYPRODUCTS ${PADDLE2ONNX_COMPILE_LIB})

add_library(external_paddle2onnx STATIC IMPORTED GLOBAL)
set_property(TARGET external_paddle2onnx PROPERTY IMPORTED_LOCATION
                                         ${PADDLE2ONNX_COMPILE_LIB})
add_dependencies(external_paddle2onnx ${PADDLE2ONNX_PROJECT})
