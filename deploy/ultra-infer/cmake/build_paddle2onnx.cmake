add_definitions(-DMAX_ONNX_OPSET_VERSION=16)
add_definitions(-DPADDLE2ONNX_LIB)

# Third dependency: onnx
if(NOT TARGET onnx_proto)
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "paddle2onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")

  set(MSVC_STATIC_CRT ON)
  if(ONNX_CUSTOM_PROTOC_PATH)
    if(WIN32)
      if(MSVC_STATIC_CRT)
        # MT
        set(ONNX_USE_MSVC_STATIC_RUNTIME ON)
      else()
        # MD
        set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
      endif()
      set(ONNX_CUSTOM_PROTOC_PATH "${ONNX_CUSTOM_PROTOC_PATH};$ENV{PATH}")
    else()
      set(ONNX_CUSTOM_PROTOC_PATH "${ONNX_CUSTOM_PROTOC_PATH}:$ENV{PATH}")
    endif()
    set(ENV{PATH} ${ONNX_CUSTOM_PROTOC_PATH})
  endif()

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/onnx)

include_directories(${PROJECT_SOURCE_DIR}/third_party/optimizer)
add_subdirectory(${PROJECT_SOURCE_DIR}/paddle2onnx/proto)

file(GLOB_RECURSE PADDLE2ONNX_ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/*.cc ${PROJECT_SOURCE_DIR}/third_party/optimizer/onnxoptimizer/*.cc)
list(REMOVE_ITEM PADDLE2ONNX_ALL_SRCS  ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc ${PROJECT_SOURCE_DIR}/third_party/optimizer/onnxoptimizer/cpp2py_export.cc)
