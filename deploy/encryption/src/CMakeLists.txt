set(SRC_COMMON
        util/crypto/basic.cpp
        util/system_utils.cpp
        util/io_utils.cpp
        util/crypto/aes_gcm.cpp
        util/crypto/sha256_utils.cpp
        util/crypto/base64_utils.cpp)

set(SRC_ENCRYPT
        safeapi/paddle_model_encrypt.cpp
        ${SRC_COMMON}
        )

set(SRC_DECRYPT
        safeapi/paddle_model_decrypt.cpp
        ${SRC_COMMON})

# encrypt: libpmodel-encrypt.so
add_library(pmodel-encrypt SHARED
        ${SRC_ENCRYPT})
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_libraries(pmodel-decrypt
                ${OPENSSL_LIBS}
                )
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	target_link_libraries(pmodel-encrypt
        	${OPENSSL_LIBS}
		-ldl -lpthread
		)
ENDIF ()

# decrypt: libpmodel-decrypt.so
add_library(pmodel-decrypt SHARED
        ${SRC_DECRYPT})
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_libraries(pmodel-decrypt
                ${OPENSSL_LIBS}
                ${PADDLE_LIBS}
                )
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(pmodel-decrypt
                ${OPENSSL_LIBS}
		-ldl -lpthread
		)
ENDIF ()


# tool: paddle_encrypt_tool
add_executable(paddle_encrypt_tool
        ../sample/paddle_encrypt_tool.cpp
        safeapi/paddle_model_encrypt.cpp
        util/crypto/basic.cpp
        util/system_utils.cpp
        util/io_utils.cpp
        util/crypto/aes_gcm.cpp
        util/crypto/sha256_utils.cpp
        util/crypto/base64_utils.cpp)
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_libraries(pmodel-decrypt
                ${OPENSSL_LIBS}
                )
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	target_link_libraries(paddle_encrypt_tool
        	${OPENSSL_LIBS}
		-ldl -lpthread
        	)
ENDIF ()
