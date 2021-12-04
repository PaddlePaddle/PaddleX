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
    set(DEPS ${DEPS} ${PADDLE_DIR}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} ${PADDLE_DIR}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
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

#set GPU
if (WITH_PADDLE_TENSORRT AND WITH_GPU)
  include_directories("${TENSORRT_DIR}/include")
  link_directories("${TENSORRT_DIR}/lib")

  file(READ ${TENSORRT_DIR}/include/NvInfer.h TENSORRT_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION
    "${TENSORRT_VERSION_FILE_CONTENTS}")
  if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
    file(READ ${TENSORRT_DIR}/include/NvInferVersion.h TENSORRT_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION
      "${TENSORRT_VERSION_FILE_CONTENTS}")
  endif()
  if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
    message(SEND_ERROR "Failed to detect TensorRT version.")
  endif()
  string(REGEX REPLACE "define NV_TENSORRT_MAJOR +([0-9]+)" "\\1"
    TENSORRT_MAJOR_VERSION "${TENSORRT_MAJOR_VERSION}")
  message(STATUS "Current TensorRT header is ${TENSORRT_INCLUDE_DIR}/NvInfer.h. "
    "Current TensorRT version is v${TENSORRT_MAJOR_VERSION}. ")
endif()

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

    if (WITH_PADDLE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()

  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})

    if (WITH_PADDLE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
      if(${TENSORRT_MAJOR_VERSION} GREATER_EQUAL 7)
        set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/myelin64_1${CMAKE_STATIC_LIBRARY_SUFFIX})
      endif()
    endif()
  endif()
endif()

message("-----DEPS = ${DEPS}")

# engine src
set(ENGINE_SRC ${PROJECT_SOURCE_DIR}/model_deploy/engine/src/ppinference_engine.cpp)

ADD_library(model_infer SHARED ${PROJECT_SOURCE_DIR}/demo/model_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC} ${ENCRYPTION_SRC})
# add_executable(model_infer model_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC} ${ENCRYPTION_SRC})
ADD_DEPENDENCIES(model_infer ext-yaml-cpp)
target_link_libraries(model_infer ${DEPS})

add_executable(batch_infer batch_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC} ${ENCRYPTION_SRC})
ADD_DEPENDENCIES(batch_infer ext-yaml-cpp)
target_link_libraries(batch_infer ${DEPS})

add_executable(multi_gpu_model_infer multi_gpu_model_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC} ${ENCRYPTION_SRC})
ADD_DEPENDENCIES(multi_gpu_model_infer ext-yaml-cpp)
target_link_libraries(multi_gpu_model_infer ${DEPS})

if (WITH_PADDLE_TENSORRT)
  add_executable(tensorrt_infer tensorrt_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC} ${ENCRYPTION_SRC})
  ADD_DEPENDENCIES(tensorrt_infer ext-yaml-cpp)
  target_link_libraries(tensorrt_infer ${DEPS})
endif()

if(WIN32)
  add_custom_command(TARGET model_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_DIR}/third_party/install/mklml/lib/mklml.dll ${CMAKE_BINARY_DIR}/paddle_deploy
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_DIR}/third_party/install/mklml/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/paddle_deploy
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_DIR}/third_party/install/mkldnn/lib/mkldnn.dll  ${CMAKE_BINARY_DIR}/paddle_deploy
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_DIR}/paddle/lib/paddle_inference.dll ${CMAKE_BINARY_DIR}/paddle_deploy
  )
  if (WITH_PADDLE_TENSORRT)
    add_custom_command(TARGET model_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_DIR}/lib/nvinfer.dll ${CMAKE_BINARY_DIR}/paddle_deploy
      COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_DIR}/lib/nvinfer_plugin.dll ${CMAKE_BINARY_DIR}/paddle_deploy
    )
    if(${TENSORRT_MAJOR_VERSION} GREATER_EQUAL 7)
      add_custom_command(TARGET model_infer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_DIR}/lib/myelin64_1.dll ${CMAKE_BINARY_DIR}/paddle_deploy
      )
    endif()
  endif()
endif()
