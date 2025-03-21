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

set(COMPRESSED_SUFFIX ".tgz")

if(WIN32)
  if(NOT CMAKE_CL_64)
    set(OPENCV_FILENAME "opencv-win-x86-3.4.16")
  else()
    set(OPENCV_FILENAME "opencv-win-x64-3.4.16")
  endif()
  set(COMPRESSED_SUFFIX ".zip")
elseif(APPLE)
  if(CURRENT_OSX_ARCH MATCHES "arm64")
    set(OPENCV_FILENAME "opencv-osx-arm64-3.4.16")
  else()
    set(OPENCV_FILENAME "opencv-osx-x86_64-3.4.16")
  endif()
elseif(IOS)
  message(FATAL_ERROR "Not support cross compiling for IOS now!")
# Linux
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_FILENAME "opencv-linux-aarch64-3.4.14")
  endif()
endif()

if(NOT OPENCV_FILENAME)
  set(OPENCV_FILENAME "opencv-linux-x64-3.4.16")
endif()

set(OPENCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/)
if(WIN32)
  if(NOT CMAKE_CL_64)
    set(OPENCV_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
  else()
    set(OPENCV_URL_PREFIX "https://bj.bcebos.com/paddle2onnx/libs")
  endif()
else() # TODO: use ultra_infer/third_libs instead.
  set(OPENCV_URL_PREFIX "https://bj.bcebos.com/paddle2onnx/libs")
endif()
if(NOT OPENCV_URL)
  set(OPENCV_URL ${OPENCV_URL_PREFIX}/${OPENCV_FILENAME}${COMPRESSED_SUFFIX})
endif()


if(BUILD_ON_JETSON)
  if(EXISTS /usr/lib/aarch64-linux-gnu/cmake/opencv4/)
    set(OPENCV_DIRECTORY /usr/lib/aarch64-linux-gnu/cmake/opencv4/)
  endif()
endif()

if(OPENCV_DIRECTORY)
  message(STATUS "Use the opencv lib specified by user. The OpenCV path: ${OPENCV_DIRECTORY}")
  STRING(REGEX REPLACE "\\\\" "/" OPENCV_DIRECTORY ${OPENCV_DIRECTORY})
  # Win/Linux/Mac
  set(OpenCV_DIR ${OPENCV_DIRECTORY})
  find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
else()
  message(STATUS "Use the default OpenCV lib from: ${OPENCV_URL}")
  # Win/Linux/Mac
  download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_FILENAME}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH}/install/)
  if(EXISTS ${THIRD_PARTY_PATH}/install/opencv)
    file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/opencv) 
  endif()
  file(RENAME ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME}/ ${THIRD_PARTY_PATH}/install/opencv)
  set(OPENCV_FILENAME opencv)
  if(NOT OpenCV_DIR)
    set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME})
  endif()
  if (WIN32)
    set(OpenCV_DIR ${OpenCV_DIR}/build)
  endif()
  find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND DEPEND_LIBS opencv_core opencv_video opencv_highgui opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_flann)
endif()
