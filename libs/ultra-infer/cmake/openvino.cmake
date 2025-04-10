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

if (OPENVINO_DIRECTORY)
  message(STATUS "Use the openvino lib specified by user. The OpenVINO path: ${OPENVINO_DIRECTORY}")
  STRING(REGEX REPLACE "\\\\" "/" OPENVINO_DIRECTORY ${OPENVINO_DIRECTORY})
  get_openvino_libs(${OPENVINO_DIRECTORY}/runtime)
  include_directories(${OPENVINO_DIRECTORY}/runtime/include ${OPENVINO_DIRECTORY}/runtime/include/ie)
  set(OPENVINO_LIB_DIR
    "${OPENVINO_DIRECTORY}/runtime/lib/intel64"
    CACHE PATH "openvino lib directory." FORCE)
  set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${OPENVINO_LIB_DIR}")

else()
  set(OPENVINO_PROJECT "extern_openvino")

  set(OPENVINO_VERSION "2022.2.0.dev20220829")
  set(OPENVINO_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs/")

  set(COMPRESSED_SUFFIX ".tgz")
  if(WIN32)
    set(OPENVINO_FILENAME "w_openvino_toolkit_windows_${OPENVINO_VERSION}")
    set(COMPRESSED_SUFFIX ".zip")
    if(NOT CMAKE_CL_64)
      message(FATAL_ERROR "UltraInfer cannot ENABLE_OPENVINO_BACKEND in win32 now.")
    endif()
  elseif(APPLE)
    if(CURRENT_OSX_ARCH MATCHES "arm64")
      message("Cannot compile with openvino while in osx arm64 platform right now")
    else()
      set(OPENVINO_FILENAME "m_openvino_toolkit_osx_${OPENVINO_VERSION}")
    endif()
  else()
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      message("Cannot compile with openvino while in linux-aarch64 platform")
    else()
      set(OPENVINO_VERSION "dev.2023.03.2")
      if(NEED_ABI0)
        set(OPENVINO_FILENAME "openvino-linux-x64-20230302-abi0")
      else()
        set(OPENVINO_VERSION "2025.0")
        set(OPENVINO_FILENAME "openvino-linux-x64-${OPENVINO_VERSION}")
      endif()
    endif()
  endif()
  set(OPENVINO_URL "${OPENVINO_URL_PREFIX}${OPENVINO_FILENAME}${COMPRESSED_SUFFIX}")

  download_and_decompress(${OPENVINO_URL}
      ${CMAKE_CURRENT_BINARY_DIR}/${OPENVINO_FILENAME}${COMPRESSED_SUFFIX}
      ${THIRD_PARTY_PATH}/install)

  if(EXISTS ${THIRD_PARTY_PATH}/install/openvino)
    file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/openvino) 
  endif()

  file(RENAME ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME} ${THIRD_PARTY_PATH}/install/openvino)
  set(OPENVINO_FILENAME openvino)

  set(OPENVINO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/runtime)
  set(OPENVINO_INSTALL_INC_DIR
    "${OPENVINO_INSTALL_DIR}/include"
    "${OPENVINO_INSTALL_DIR}/include/ie"
    CACHE PATH "openvino install include directory." FORCE)
    
  set(OPENVINO_LIB_DIR
    "${OPENVINO_INSTALL_DIR}/lib/"
    "${OPENVINO_INSTALL_DIR}/3rdparty/tbb/lib/"
    CACHE PATH "openvino lib directory." FORCE)
  set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${OPENVINO_LIB_DIR}")

  # For OPENVINO code to include internal headers.
  include_directories(${OPENVINO_INSTALL_INC_DIR})

  if(WIN32)
    file(GLOB_RECURSE OPENVINO_LIB_FILES ${OPENVINO_INSTALL_DIR}/lib/intel64/Release/*)
    file(COPY ${OPENVINO_LIB_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/lib/)
    file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/lib/intel64)

    file(GLOB_RECURSE OPENVINO_BIN_FILES ${OPENVINO_INSTALL_DIR}/bin/intel64/Release/*)
    file(COPY ${OPENVINO_BIN_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/bin/)
    file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/bin/intel64)
  elseif(APPLE)
    file(GLOB_RECURSE OPENVINO_LIB_FILES ${OPENVINO_INSTALL_DIR}/lib/intel64/Release/*)
    file(COPY ${OPENVINO_LIB_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/lib/)
    file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/lib/intel64)
  else()
    file(GLOB_RECURSE OPENVINO_LIB_FILES ${OPENVINO_INSTALL_DIR}/lib/intel64/*)
    file(COPY ${OPENVINO_LIB_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/lib/)
    file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/lib/intel64)
  endif()

  file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/docs)
  file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/install_dependencies)
  file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/samples)
  file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/setupvars.sh)
  file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/tools)
  get_openvino_libs(${OPENVINO_INSTALL_DIR})
endif()
message("OPENVINO_LIBS = ${OPENVINO_LIBS}")
list(APPEND DEPEND_LIBS ${OPENVINO_LIBS})
