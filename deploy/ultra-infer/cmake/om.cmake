set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()


set(NPU_libs ascendcl stdc++)

include_directories(
   ${INC_PATH}/runtime/include/
)

link_directories(
    ${LIB_PATH}
)
