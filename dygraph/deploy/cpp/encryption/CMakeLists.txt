cmake_minimum_required(VERSION 3.12)
project(paddle_model_protect)

set(CMAKE_CXX_STANDARD 11)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        # use "-fvisibility=hidden" instead of "-Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/export_rule.map"
        set(CMAKE_C_VISIBILITY_PRESET hidden)
        set(CMAKE_CXX_VISIBILITY_PRESET hidden)
        set(CMAKE_C_FLAGS "-g -O2 -fPIC ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-g -O2 -fPIC ${CMAKE_CXX_FLAGS}")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(CMAKE_C_FLAGS_RELEASE "/MT")
        set(CMAKE_CXX_FLAGS_RELEASE "/MT")
ENDIF ()

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output/lib)

file(COPY "${PROJECT_SOURCE_DIR}/include/paddle_model_encrypt.h" DESTINATION "${PROJECT_SOURCE_DIR}/output/include/paddle_model_encrypt.h")
file(COPY "${PROJECT_SOURCE_DIR}/include/paddle_model_decrypt.h" DESTINATION "${PROJECT_SOURCE_DIR}/output/include/paddle_model_decrypt.h")
file(COPY "${PROJECT_SOURCE_DIR}/include/paddle_stream_decrypt.h" DESTINATION "${PROJECT_SOURCE_DIR}/output/include/paddle_stream_decrypt.h")
file(COPY "${PROJECT_SOURCE_DIR}/include/model_code.h" DESTINATION "${PROJECT_SOURCE_DIR}/output/include/model_code.h")

set(OPENSSL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/openssl-1.1.0k/install-${CMAKE_SYSTEM_PROCESSOR}")
include_directories("." include ${OPENSSL_ROOT_DIR}/include)
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(OPENSSL_LIBS
                "${OPENSSL_ROOT_DIR}/lib/libssl_static.lib"
                "${OPENSSL_ROOT_DIR}/lib/libcrypto_static.lib")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(OPENSSL_LIBS
                "${OPENSSL_ROOT_DIR}/lib/libssl.a"
                "${OPENSSL_ROOT_DIR}/lib/libcrypto.a"
                -ldl -lpthread)
ENDIF ()

message("build no paddle libs")
IF (MSVC)
        # Visual Studio 2015
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ws2_32.lib /NODEFAULTLIB:libcmt.lib")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ws2_32.lib /NODEFAULTLIB:libcmt.lib")
ENDIF ()

aux_source_directory(util/src SRC_COMMONC)
set(SRC_ENCRYPT src/paddle_model_encrypt.cpp ${SRC_COMMON})

# tool: paddle_encrypt_tool
add_executable(paddle_encrypt_tool
        sample/paddle_encrypt_tool.cpp
        ${SRC_ENCRYPT})
target_link_libraries(paddle_encrypt_tool ${OPENSSL_LIBS})
