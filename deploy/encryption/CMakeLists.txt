cmake_minimum_required(VERSION 3.12)
project(paddle_model_protect)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "-g -O2 -fPIC -lpthread -ldl -Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/export_rule.map ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-g -O2 -fPIC -lpthread -ldl -Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/export_rule.map ${CMAKE_CXX_FLAGS}")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output/lib)

file(COPY "${PROJECT_SOURCE_DIR}/include/paddle_model_encrypt.h" DESTINATION "${PROJECT_SOURCE_DIR}/output/include/paddle_model_encrypt.h")
file(COPY "${PROJECT_SOURCE_DIR}/include/paddle_model_decrypt.h" DESTINATION "${PROJECT_SOURCE_DIR}/output/include/paddle_model_decrypt.h")
file(COPY "${PROJECT_SOURCE_DIR}/include/model_code.h" DESTINATION "${PROJECT_SOURCE_DIR}/output/include/model_code.h")

set(SRC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(OPENSSL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/openssl-1.1.0k/install-${CMAKE_SYSTEM_PROCESSOR}")
set(OPENSSL_INCLUDE "${OPENSSL_ROOT_DIR}/include")
set(OPENSSL_LIBS
        "${OPENSSL_ROOT_DIR}/lib/libssl.a"
        "${OPENSSL_ROOT_DIR}/lib/libcrypto.a")

include_directories(
        include
        ${OPENSSL_INCLUDE}
)

add_subdirectory(${SRC_ROOT_DIR})
