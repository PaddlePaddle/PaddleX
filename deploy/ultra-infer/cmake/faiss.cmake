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

set(FAISS_PROJECT external_faiss_download)
set(FAISS_FILENAME faiss)  
set(FAISS_PREFIX_DIR ${THIRD_LIBS_PATH}/${FAISS_FILENAME})
set(FAISS_SOURCE_DIR ${THIRD_LIBS_PATH}/${FAISS_FILENAME}/src/${FAISS_PROJECT})
set(FAISS_INSTALL_DIR ${THIRD_LIBS_PATH}/install/${FAISS_FILENAME})
set(FAISS_INC_DIR ${FAISS_INSTALL_DIR}/include CACHE PATH "faiss include directory." FORCE)
set(FAISS_LIB_DIR ${FAISS_INSTALL_DIR}/lib CACHE PATH "faiss lib directory." FORCE)

if(NOT WITH_FAISS_STATIC)
  message(FATAL_ERROR "Not support WITH_FAISS_STATIC=OFF now!")
endif()

set(FAISS_URL_PREFIX "https://bj.bcebos.com/fastdeploy/test")

set(FAISS_VERSION 1.7.3)
# URL
if(NOT FAISS_URL)
  if(WIN32)
    set(FAISS_URL "${FAISS_URL_PREFIX}/faiss-win-x64-${FAISS_VERSION}.zip")
  elseif(APPLE)
    if(CURRENT_OSX_ARCH MATCHES "arm64")
      set(FAISS_URL "${FAISS_URL_PREFIX}/faiss-osx-arm64-${FAISS_VERSION}.tgz")
    else()
      set(FAISS_URL "${FAISS_URL_PREFIX}/faiss-osx-x64-${FAISS_VERSION}.tgz")
    endif()  
  else() # Linux
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      message(FATAL_ERROR "Not support for Linux aarch64 now!")
    else()
      if(WITH_FAISS_GPU)
        set(FAISS_URL "${FAISS_URL_PREFIX}/faiss-linux-x64-gpu-${FAISS_VERSION}.tgz")
      else()
        set(FAISS_URL "${FAISS_URL_PREFIX}/faiss-linux-x64-${FAISS_VERSION}.tgz")
      endif()
    endif()
  endif()
endif()

# FAISS Headers
include_directories(${FAISS_INC_DIR})

# FAISS Libs paths
if(WIN32)
  set(FAISS_LIB "${FAISS_LIB_DIR}/faiss.lib")
elseif(APPLE)
  set(FAISS_LIB "${FAISS_LIB_DIR}/libfaiss.a")
else() # Linux
  set(FAISS_LIB "${FAISS_LIB_DIR}/libfaiss.a")
endif()

# Download FAISS
ExternalProject_Add(
  ${FAISS_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${FAISS_URL}
  PREFIX ${FAISS_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E remove_directory ${FAISS_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E make_directory ${FAISS_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E rename ${FAISS_SOURCE_DIR}/lib/ ${FAISS_INSTALL_DIR}/lib &&
    ${CMAKE_COMMAND} -E copy_directory ${FAISS_SOURCE_DIR}/include ${FAISS_INC_DIR}
  BUILD_BYPRODUCTS ${FAISS_LIB})

set(FAISS_LIBRARIES)

add_library(external_faiss STATIC IMPORTED GLOBAL)
set_property(TARGET external_faiss PROPERTY IMPORTED_LOCATION ${FAISS_LIB})
add_dependencies(external_faiss ${FAISS_PROJECT})

list(APPEND FAISS_LIBRARIES external_faiss)

# Add BLAS/LAPACK/OpenBLAS (needed by FAISS)
if(WIN32)
  add_library(external_blas STATIC IMPORTED GLOBAL)
  set_property(TARGET external_blas PROPERTY IMPORTED_LOCATION ${FAISS_LIB_DIR}/BLAS.lib)
  add_dependencies(external_blas ${FAISS_PROJECT})
  list(APPEND FAISS_LIBRARIES external_blas)

  add_library(external_lapack STATIC IMPORTED GLOBAL)
  set_property(TARGET external_lapack PROPERTY IMPORTED_LOCATION ${FAISS_LIB_DIR}/LAPACK.lib)
  add_dependencies(external_lapack ${FAISS_PROJECT})
  list(APPEND FAISS_LIBRARIES external_lapack)
elseif(APPLE)
  find_package(BLAS REQUIRED)
  list(APPEND FAISS_LIBRARIES ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  list(APPEND FAISS_LIBRARIES ${LAPACK_LIBRARIES})
else() # Linux
  find_package(BLAS REQUIRED)
  list(APPEND FAISS_LIBRARIES ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  list(APPEND FAISS_LIBRARIES ${LAPACK_LIBRARIES})
endif()

# Add OpenMP (REQUIRED), OpenMP must be available.
find_package(OpenMP REQUIRED)
list(APPEND FAISS_LIBRARIES OpenMP::OpenMP_CXX)

set(FAISS_INCLUDE_DIRS ${FAISS_INC_DIR})
set(FAISS_LIBS ${FAISS_LIBRARIES})
set(FAISS_FOUND TRUE)
