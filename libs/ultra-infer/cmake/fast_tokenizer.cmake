

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

set(FASTTOKENIZER_PROJECT "extern_fast_tokenizer")
set(FASTTOKENIZER_PREFIX_DIR ${THIRD_PARTY_PATH}/fast_tokenizer)
set(FASTTOKENIZER_SOURCE_DIR
    ${THIRD_PARTY_PATH}/fast_tokenizer/src/${FASTTOKENIZER_PROJECT})
set(FASTTOKENIZER_INSTALL_DIR ${THIRD_PARTY_PATH}/install/fast_tokenizer)
set(FASTTOKENIZER_INC_DIR
    "${FASTTOKENIZER_INSTALL_DIR}/include"
    "${FASTTOKENIZER_INSTALL_DIR}/third_party/include"
    CACHE PATH "fast_tokenizer include directory." FORCE)
set(FASTTOKENIZER_LIB_DIR
    "${FASTTOKENIZER_INSTALL_DIR}/lib/"
    CACHE PATH "fast_tokenizer lib directory." FORCE)

set(FASTTOKENIZER_THIRD_LIB_DIR
    "${FASTTOKENIZER_INSTALL_DIR}/third_party/lib/"
    CACHE PATH "fast_tokenizer lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${FASTTOKENIZER_LIB_DIR}")

include_directories(${FASTTOKENIZER_INC_DIR})

# Set lib path
if(WIN32)
  set(FASTTOKENIZER_COMPILE_LIB "${FASTTOKENIZER_LIB_DIR}/core_tokenizers.lib"
      CACHE FILEPATH "fast_tokenizer compile library." FORCE)
  set(ICUDT_LIB "${FASTTOKENIZER_THIRD_LIB_DIR}/icudt.lib")
  set(ICUUC_LIB "${FASTTOKENIZER_THIRD_LIB_DIR}/icuuc.lib")
elseif(APPLE)
  set(FASTTOKENIZER_COMPILE_LIB "${FASTTOKENIZER_LIB_DIR}/libcore_tokenizers.dylib"
      CACHE FILEPATH "fast_tokenizer compile library." FORCE)
else()
  set(FASTTOKENIZER_COMPILE_LIB "${FASTTOKENIZER_LIB_DIR}/libcore_tokenizers.so"
      CACHE FILEPATH "fast_tokenizer compile library." FORCE)
endif(WIN32)
message("FASTTOKENIZER_COMPILE_LIB = ${FASTTOKENIZER_COMPILE_LIB}")

set(FASTTOKENIZER_URL_BASE "https://bj.bcebos.com/paddlenlp/fast_tokenizer/")
set(FASTTOKENIZER_VERSION "1.0.2")

# Set download url
if(WIN32)
  set(FASTTOKENIZER_FILE "fast_tokenizer-win-x64-${FASTTOKENIZER_VERSION}.zip")
  if(NOT CMAKE_CL_64)
    set(FASTTOKENIZER_FILE "fast_tokenizer-win-x86-${FASTTOKENIZER_VERSION}.zip")
  endif()
elseif(APPLE)
  if(CURRENT_OSX_ARCH MATCHES "arm64")
    set(FASTTOKENIZER_FILE "fast_tokenizer-osx-arm64-${FASTTOKENIZER_VERSION}.tgz")
  else()
    set(FASTTOKENIZER_FILE "fast_tokenizer-osx-x86_64-${FASTTOKENIZER_VERSION}.tgz")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(FASTTOKENIZER_FILE "fast_tokenizer-linux-aarch64-${FASTTOKENIZER_VERSION}.tgz")
  else()
    set(FASTTOKENIZER_FILE "fast_tokenizer-linux-x64-${FASTTOKENIZER_VERSION}.tgz")
  endif()
endif()
set(FASTTOKENIZER_URL "${FASTTOKENIZER_URL_BASE}${FASTTOKENIZER_FILE}")

ExternalProject_Add(
  ${FASTTOKENIZER_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${FASTTOKENIZER_URL}
  PREFIX ${FASTTOKENIZER_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${FASTTOKENIZER_SOURCE_DIR} ${FASTTOKENIZER_INSTALL_DIR}
  BUILD_BYPRODUCTS ${FASTTOKENIZER_COMPILE_LIB})

add_library(fast_tokenizer STATIC IMPORTED GLOBAL)
set_property(TARGET fast_tokenizer PROPERTY IMPORTED_LOCATION ${FASTTOKENIZER_COMPILE_LIB})
add_dependencies(fast_tokenizer ${FASTTOKENIZER_PROJECT})
list(APPEND DEPEND_LIBS fast_tokenizer)

if (WIN32)
  add_library(icudt STATIC IMPORTED GLOBAL)
  set_property(TARGET icudt PROPERTY IMPORTED_LOCATION ${ICUDT_LIB})
  add_dependencies(icudt ${FASTTOKENIZER_PROJECT})
  list(APPEND DEPEND_LIBS icudt)

  add_library(icuuc STATIC IMPORTED GLOBAL)
  set_property(TARGET icuuc PROPERTY IMPORTED_LOCATION ${ICUUC_LIB})
  add_dependencies(icuuc ${FASTTOKENIZER_PROJECT})
  list(APPEND DEPEND_LIBS icuuc)
endif()
