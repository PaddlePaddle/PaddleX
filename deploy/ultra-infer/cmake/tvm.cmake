# set path

set(TVM_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
set(TVM_VERSION "0.12.0")
set(TVM_SYSTEM "")

if (${CMAKE_SYSTEM} MATCHES "Darwin")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
        set(TVM_SYSTEM "macos-arm64")
    endif ()
elseif (${CMAKE_SYSTEM} MATCHES "Linux")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86")
        set(TVM_SYSTEM "linux-x86")
    endif ()
else ()
    error("TVM only support MacOS in Arm64 or linux in x86")
endif ()
set(TVM_FILE "tvm-${TVM_SYSTEM}-${TVM_VERSION}.tgz")
set(TVM_URL "${TVM_URL_BASE}${TVM_FILE}")

set(TVM_RUNTIME_PATH "${THIRD_PARTY_PATH}/install/tvm")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${TVM_RUNTIME_PATH}")
download_and_decompress(${TVM_URL}
        "${CMAKE_CURRENT_BINARY_DIR}/${TVM_FILE}"
        "${THIRD_PARTY_PATH}/install/")
include_directories(${TVM_RUNTIME_PATH}/include)

# copy dlpack to third_party
set(DLPACK_PATH "${THIRD_PARTY_PATH}/install/dlpack")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${DLPACK_PATH}")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/third_party/dlpack"
        "${THIRD_PARTY_PATH}/install/dlpack")
include_directories(${DLPACK_PATH}/include)

set(DMLC_CORE_PATH "${THIRD_PARTY_PATH}/install/dmlc-core")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${DMLC_CORE_PATH}")
set(DMLC_CORE_URL https://bj.bcebos.com/fastdeploy/third_libs/dmlc-core.tgz)
download_and_decompress(${DMLC_CORE_URL}
        "${CMAKE_CURRENT_BINARY_DIR}/dmlc-core.tgz"
        "${THIRD_PARTY_PATH}/install/")
include_directories(${DMLC_CORE_PATH}/include)

# include lib
if (EXISTS ${TVM_RUNTIME_PATH})
    if (${CMAKE_SYSTEM} MATCHES "Darwin")
        set(TVM_RUNTIME_LIB ${TVM_RUNTIME_PATH}/lib/libtvm_runtime.dylib)
    elseif (${CMAKE_SYSTEM} MATCHES "Linux")
        set(TVM_RUNTIME_LIB ${TVM_RUNTIME_PATH}/lib/libtvm_runtime.so)
    endif ()
    include(${TVM_RUNTIME_PATH}/lib/cmake/tvm/tvmConfig.cmake)
    add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
else ()
    error(FATAL_ERROR "[tvm.cmake] TVM_RUNTIME_PATH does not exist.")
endif ()
