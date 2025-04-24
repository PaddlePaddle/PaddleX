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

function(download_patchelf)
  if(UNIX AND (NOT APPLE))
    set(PATCHELF_EXE "patchelf")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/patchelf-0.15.0-aarch64.tar.gz)
      download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/patchelf-0.15.0-aarch64.tar.gz ${THIRD_PARTY_PATH}/patchelf)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/patchelf-0.15.0-x86_64.tar.gz)
      download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/patchelf-0.15.0-x86_64.tar.gz ${THIRD_PARTY_PATH}/patchelf)
    endif()
  endif()
endfunction()

function(download_protobuf)
  if(WIN32)
    if(NOT CMAKE_CL_64)
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-win-x86-3.16.0.zip)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-win-x64-3.16.0.zip)
    endif()
    set(ORIGIN_ENV_PATH "$ENV{PATH}")
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/protobuf-win-3.16.0.tgz ${THIRD_PARTY_PATH}/protobuf)
    set(ENV{PATH} "${THIRD_PARTY_PATH}\\protobuf\\bin;${ORIGIN_ENV_PATH}")
  elseif(APPLE)
    if(CURRENT_OSX_ARCH MATCHES "arm64")
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-osx-arm64-3.16.0.tgz)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-osx-x86_64-3.16.0.tgz)
    endif()
    set(ORIGIN_ENV_PATH "$ENV{PATH}")
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/protobuf-osx-3.16.0.tgz ${THIRD_PARTY_PATH}/protobuf)
    set(ENV{PATH} "${THIRD_PARTY_PATH}/protobuf/bin/:${ORIGIN_ENV_PATH}")
  else()
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-linux-aarch64-3.16.0.tgz)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-linux-x64-3.16.0.tgz)
    endif()
    set(ORIGIN_ENV_PATH "$ENV{PATH}")
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/protobuf-linux-3.16.0.tgz ${THIRD_PARTY_PATH}/protobuf)
    set(ENV{PATH} "${THIRD_PARTY_PATH}/protobuf/bin/:${ORIGIN_ENV_PATH}")
  endif()
endfunction()

function(download_eigen)
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_party/eigen-linux-x86-241210.tgz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/eigen-linux-x86-241210.tgz ${THIRD_PARTY_DIR}/eigen)
endfunction()

function(download_yaml_cpp)
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_party/yaml-cpp-linux-x86-241210.tgz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp-linux-x86-241210.tgz ${THIRD_PARTY_DIR}/yaml-cpp)
endfunction()

function(download_pybind)
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_party/pybind11-linux-x86-241210.tgz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/pybind11-linux-x86-241210.tgz ${THIRD_PARTY_DIR}/pybind11)
endfunction()

function(download_dlpack)
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_party/dlpack-linux-x86-241210.tgz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/dlpack-linux-x86-241210.tgz ${THIRD_PARTY_DIR}/dlpack)
endfunction()

function(download_onnx)
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_party/onnx-linux-x86-241210.tgz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/onnx-linux-x86-241210.tgz ${THIRD_PARTY_DIR}/onnx)
endfunction()

function(download_optimizer)
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_party/optimizer-linux-x86-241210.tgz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/optimizer-linux-x86-241210.tgz ${THIRD_PARTY_DIR}/optimizer)
endfunction()
