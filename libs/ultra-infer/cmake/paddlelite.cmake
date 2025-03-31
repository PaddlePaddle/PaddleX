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

option(PADDLELITE_DIRECTORY "Directory of custom Paddle-Lite library" OFF)

set(PADDLELITE_PROJECT "extern_paddlelite")
set(PADDLELITE_FILENAME paddlelite)
set(PADDLELITE_PREFIX_DIR ${THIRD_PARTY_PATH}/${PADDLELITE_FILENAME})
set(PADDLELITE_SOURCE_DIR
    ${THIRD_PARTY_PATH}/${PADDLELITE_FILENAME}/src/${PADDLELITE_PROJECT})
set(PADDLELITE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${PADDLELITE_FILENAME})  
set(PADDLELITE_INC_DIR
    "${PADDLELITE_INSTALL_DIR}/include"
    CACHE PATH "paddlelite include directory." FORCE)
set(PADDLELITE_LIB_DIR
"${PADDLELITE_INSTALL_DIR}/lib/"
CACHE PATH "paddlelite lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${PADDLELITE_LIB_DIR}")

if(PADDLELITE_DIRECTORY)
  set(PADDLELITE_INC_DIR "${PADDLELITE_DIRECTORY}/include")
endif()

include_directories(${PADDLELITE_INC_DIR})

if(PADDLELITE_DIRECTORY)
  # Use custom Paddle-Lite libs.
  if(EXISTS "${THIRD_PARTY_PATH}/install/paddlelite")
    file(REMOVE_RECURSE "${THIRD_PARTY_PATH}/install/paddlelite")
  endif()
  if(WIN32 OR APPLE OR IOS)
    message(FATAL_ERROR "Doesn't support windows/mac/ios platform with backend Paddle Lite now.")
  else()
    message(STATUS "Copying ${PADDLELITE_DIRECTORY} to ${THIRD_PARTY_PATH}/install/paddlelite ...")
    execute_process(COMMAND mkdir -p ${THIRD_PARTY_PATH}/install/paddlelite)
    execute_process(COMMAND cp -r ${PADDLELITE_DIRECTORY}/include ${THIRD_PARTY_PATH}/install/paddlelite)
    execute_process(COMMAND mkdir -p ${PADDLELITE_LIB_DIR})
    execute_process(COMMAND cp -r ${PADDLELITE_DIRECTORY}/lib/ ${PADDLELITE_LIB_DIR})
    message(${PADDLELITE_LIB_DIR})
    execute_process(COMMAND rm -rf ${PADDLELITE_LIB_DIR}/*.a)
    set(PADDLELITE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_full_api_shared.so")
  endif()

else()
  # Use default Paddle-Lite libs.
  set(PADDLELITE_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
  
  if(WIN32 OR APPLE OR IOS)
    message(FATAL_ERROR "Doesn't support windows/mac/ios platform with backend Paddle Lite now.")
  else()
    set(PADDLELITE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_full_api_shared.so")
    set(PADDLELITE_REMOVE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_api_full_bundled.a")
  endif()

  if(NOT PADDLELITE_URL)
    if(WIN32 OR APPLE OR IOS)
      message(FATAL_ERROR "Doesn't support windows/mac/ios platform with backend Paddle Lite now.")
    else() # Linux
      if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(PADDLELITE_URL "${PADDLELITE_URL_PREFIX}/lite-linux-arm64-20221209.tgz")
        set(PADDLELITE_VERSION 0.0.0.20221209)
      else()
        message(FATAL_ERROR "Only support Linux aarch64 now, x64 is not supported with backend Paddle Lite.")
      endif()
    endif()
  endif()

  ExternalProject_Add(
    ${PADDLELITE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLELITE_URL}
    PREFIX ${PADDLELITE_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${PADDLELITE_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${PADDLELITE_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E rename ${PADDLELITE_SOURCE_DIR}/lib/ ${PADDLELITE_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E copy_directory ${PADDLELITE_SOURCE_DIR}/include ${PADDLELITE_INC_DIR}
    BUILD_BYPRODUCTS ${PADDLELITE_LIB})

endif(PADDLELITE_DIRECTORY)

if(UNIX AND (NOT APPLE) AND BUILD_ULTRAINFER_PYTHON)
  add_custom_target(patchelf_paddle_lite ALL COMMAND  bash -c "PATCHELF_EXE=${PATCHELF_EXE} python ${PROJECT_SOURCE_DIR}/scripts/patch_paddle_lite.py ${PADDLELITE_INSTALL_DIR}/lib/" DEPENDS ${LIBRARY_NAME})
endif()

add_library(external_paddle_lite STATIC IMPORTED GLOBAL)
set_property(TARGET external_paddle_lite PROPERTY IMPORTED_LOCATION ${PADDLELITE_LIB})
add_dependencies(external_paddle_lite ${PADDLELITE_PROJECT})
