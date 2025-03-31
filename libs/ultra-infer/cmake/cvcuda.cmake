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

if(NOT WITH_GPU)
  message(FATAL_ERROR "ENABLE_CVCUDA is available on Linux and WITH_GPU=ON, but now WITH_GPU=OFF.")
endif()

if(APPLE OR IOS OR WIN32)
  message(FATAL_ERROR "Cannot enable CV-CUDA in mac/ios/windows os, please set -DENABLE_CVCUDA=OFF.")
endif()

if(NOT (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64"))
  message(FATAL_ERROR "CV-CUDA only support x86_64.")
endif()

set(CVCUDA_LIB_URL https://github.com/CVCUDA/CV-CUDA/releases/download/v0.2.1-alpha/nvcv-lib-0.2.1_alpha-cuda11-x86_64-linux.tar.xz)
set(CVCUDA_LIB_FILENAME nvcv-lib-0.2.1_alpha-cuda11-x86_64-linux.tar.xz)
set(CVCUDA_DEV_URL https://github.com/CVCUDA/CV-CUDA/releases/download/v0.2.1-alpha/nvcv-dev-0.2.1_alpha-cuda11-x86_64-linux.tar.xz)
set(CVCUDA_DEV_FILENAME nvcv-dev-0.2.1_alpha-cuda11-x86_64-linux.tar.xz)

download_and_decompress(${CVCUDA_LIB_URL} ${CMAKE_CURRENT_BINARY_DIR}/${CVCUDA_LIB_FILENAME} ${THIRD_PARTY_PATH}/cvcuda)
download_and_decompress(${CVCUDA_DEV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${CVCUDA_DEV_FILENAME} ${THIRD_PARTY_PATH}/cvcuda)

execute_process(COMMAND rm -rf ${THIRD_PARTY_PATH}/install/cvcuda)
execute_process(COMMAND mkdir -p ${THIRD_PARTY_PATH}/install/cvcuda)
execute_process(COMMAND cp -r ${THIRD_PARTY_PATH}/cvcuda/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/ ${THIRD_PARTY_PATH}/install/cvcuda/lib)
execute_process(COMMAND cp -r ${THIRD_PARTY_PATH}/cvcuda/opt/nvidia/cvcuda0/include/ ${THIRD_PARTY_PATH}/install/cvcuda/include)

link_directories(${THIRD_PARTY_PATH}/install/cvcuda/lib)
include_directories(${THIRD_PARTY_PATH}/install/cvcuda/include)
