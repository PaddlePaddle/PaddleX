cmake_minimum_required(VERSION 3.0)
project(Installation
        LANGUAGES CXX C
        VERSION 1.0)

message("----this is build lib--------")

if(WIN32)
  add_definitions(-DPADDLEX_DEPLOY)
endif()

option(WITH_MKL        "Compile demo with MKL/OpenBlas support,defaultuseMKL."          ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
if (NOT WIN32)
    option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)
else()
    option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)
endif()
option(WITH_TENSORRT "Compile demo with TensorRT."   OFF)

SET(TENSORRT_DIR "" CACHE PATH "Location of libraries")
SET(PADDLE_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(CUDA_LIB "" CACHE PATH "Location of libraries")

get_filename_component(PROJECT_ROOT_DIR "." REALPATH)
SET(PROJECT_ROOT_DIR ${PROJECT_ROOT_DIR}  CACHE PATH  "root directory of project.")

SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/PaddleXManufacture" CACHE PATH  "project install" FORCE)
SET(PaddleDeploy_LIBS "" CACHE PATH "Location of libraries")
if (NOT WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demo)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddle_deploy)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddle_deploy)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddle_deploy)
endif()

#source
include_directories("${CMAKE_SOURCE_DIR}/")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")

#yaml-cpp
if(WIN32)
  SET(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "yaml build shared library.")
else()
  SET(YAML_BUILD_SHARED_LIBS ON CACHE BOOL "yaml build shared library.")
endif(WIN32)
include(${PROJECT_ROOT_DIR}/cmake/yaml-cpp.cmake)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")

#paddle inference
if (NOT DEFINED PADDLE_DIR OR ${PADDLE_DIR} STREQUAL "")
    message(FATAL_ERROR "please set PADDLE_DIR with -DPADDLE_DIR=/path/paddle_influence_dir")
endif()

#paddle inference third party
include_directories("${PADDLE_DIR}")
include_directories("${PADDLE_DIR}/third_party/install/protobuf/include")
include_directories("${PADDLE_DIR}/third_party/install/glog/include")
include_directories("${PADDLE_DIR}/third_party/install/gflags/include")
include_directories("${PADDLE_DIR}/third_party/install/xxhash/include")
include_directories("${PADDLE_DIR}/third_party/install/cryptopp/include")

link_directories("${PADDLE_DIR}/paddle/lib/")
link_directories("${PADDLE_DIR}/third_party/install/protobuf/lib")
link_directories("${PADDLE_DIR}/third_party/install/glog/lib")
link_directories("${PADDLE_DIR}/third_party/install/gflags/lib")
link_directories("${PADDLE_DIR}/third_party/install/xxhash/lib")
link_directories("${PADDLE_DIR}/third_party/install/cryptopp/lib")

if (WIN32)
  set(DEPS ${DEPS} ${PADDLE_DIR}/paddle/lib/paddle_inference.lib)
  set(DEPS ${DEPS} glog gflags_static libprotobuf xxhash cryptopp-static libyaml-cppmt shlwapi)
else()
  if (WITH_STATIC_LIB)
    set(DEPS ${PADDLE_DIR}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_DIR}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(DEPS ${DEPS} glog gflags protobuf xxhash cryptopp yaml-cpp)
endif(WIN32)

#MKL
if(WITH_MKL)
  ADD_DEFINITIONS(-DUSE_MKL)
  set(MKLML_PATH "${PADDLE_DIR}/third_party/install/mklml")
  include_directories("${MKLML_PATH}/include")
  if (WIN32)
    set(MATH_LIB ${MKLML_PATH}/lib/mklml.lib ${MKLML_PATH}/lib/libiomp5md.lib)
  else ()
    set(MATH_LIB ${MKLML_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} ${MKLML_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    execute_process(COMMAND cp -r ${MKLML_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} /usr/lib)
  endif ()
  set(MKLDNN_PATH "${PADDLE_DIR}/third_party/install/mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    if (WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    else ()
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
    endif ()
  endif()
else()
  set(MATH_LIB ${PADDLE_DIR}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

set(DEPS ${DEPS} ${MATH_LIB} ${MKLDNN_LIB})

#OPENCV
if (NOT (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
  if (NOT DEFINED OPENCV_DIR OR ${OPENCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
  endif()
endif()

if (WIN32)
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
  unset(OpenCV_DIR CACHE)
else ()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") # x86_64 aarch64
    set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    file(GLOB OpenCV_LIBS /usr/lib/aarch64-linux-gnu/libopencv_*${CMAKE_SHARED_LIBRARY_SUFFIX})
    message("OpenCV libs: ${OpenCV_LIBS}")
  else()
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
  endif()
endif ()

set(DEPS ${DEPS} ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

#set GPU
if(WITH_GPU)
  if (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
    message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda/lib64")
  endif()

  if(NOT WIN32)
    if (NOT DEFINED CUDNN_LIB)
      message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn/")
    endif()

    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})

    if (WITH_TENSORRT)
      include_directories("${TENSORRT_DIR}/include")
      link_directories("${TENSORRT_DIR}/lib")

      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()

  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})

    if (WITH_TENSORRT)
      include_directories("${TENSORRT_DIR}/include")
      link_directories("${TENSORRT_DIR}/lib")

      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/myelin64_1${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
  endif()
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} ${OpenMP_C_FLAGS}")
        set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o3 -fopenmp -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

message("-----DEPS = ${DEPS}")

#project
include_directories("${PROJECT_ROOT_DIR}")

aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/common/src SRC)
set(ENGINE_SRC ${PROJECT_ROOT_DIR}/model_deploy/engine/src/ppinference_engine.cpp)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/utils/src UTILS_SRC)

#detector seg
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppdet/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppseg/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/ppclas/src DETECTOR_SRC)
aux_source_directory(${PROJECT_ROOT_DIR}/model_deploy/paddlex/src DETECTOR_SRC)

#pipeline
aux_source_directory(${PROJECT_ROOT_DIR}/pipeline/src PIPELINE_SRC)

#shared lib
add_library (PaddleXManufacture SHARED  ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC} ${UTILS_SRC} ${PIPELINE_SRC})
add_dependencies(PaddleXManufacture ext-yaml-cpp)
target_link_libraries(PaddleXManufacture ${DEPS})

# Install
# project lib
install(TARGETS PaddleXManufacture
        EXPORT PaddleXManufactureTargets
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

# project header file
file(GLOB PROJECT_INCLUDE_FILE
          ${PROJECT_ROOT_DIR}/model_deploy
          ${PROJECT_ROOT_DIR}/pipeline
)
install(DIRECTORY ${PROJECT_INCLUDE_FILE}
        DESTINATION  ${CMAKE_INSTALL_PREFIX}/include
        PATTERN "*/src" EXCLUDE)

# find_package config
install(EXPORT PaddleXManufactureTargets
        FILE PaddleXManufactureTargets.cmake
        NAMESPACE PaddleXManufacture::
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/PaddleXManufacture
)
include(CMakePackageConfigHelpers)
configure_package_config_file(${PROJECT_ROOT_DIR}/tools/build_pipeline/Config.cmake.in
    PaddleXManufactureConfig.cmake
    INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/share/PaddleXManufacture"
)
write_basic_package_version_file(
    PaddleXManufactureConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${PROJECT_BINARY_DIR}/PaddleXManufactureConfig.cmake"
    "${PROJECT_BINARY_DIR}/PaddleXManufactureConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/share/PaddleXManufacture"
)

if(WIN32)
#paddle inference
file(GLOB THIRD_PARTY_FILE ${PADDLE_DIR}/third_party/install/gflags ${PADDLE_DIR}/paddle)
install(DIRECTORY ${THIRD_PARTY_FILE}
        DESTINATION  ${CMAKE_INSTALL_PREFIX}/third_party)

else()
# #paddle inference
file(GLOB THIRD_PARTY_FILE ${PADDLE_DIR}/third_party/install/gflags ${PADDLE_DIR}/paddle)
install(DIRECTORY ${THIRD_PARTY_FILE}
        DESTINATION  ${CMAKE_INSTALL_PREFIX}/third_party
        PATTERN "*/lib" EXCLUDE
)
endif()

#yaml-cpp
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include
                  ${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib
        DESTINATION  ${CMAKE_INSTALL_PREFIX}/third_party/yaml-cpp)

# mkl
if(WITH_MKL)
  install(DIRECTORY ${PADDLE_DIR}/third_party/install/mklml
          DESTINATION  ${CMAKE_INSTALL_PREFIX}/third_party)
  install(DIRECTORY ${PADDLE_DIR}/third_party/install/mkldnn
          DESTINATION  ${CMAKE_INSTALL_PREFIX}/third_party)
endif()
