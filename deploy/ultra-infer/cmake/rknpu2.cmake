# get RKNPU2_URL
set(RKNPU2_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
set(RKNPU2_VERSION "1.4.2b0")
set(RKNPU2_FILE "rknpu2_runtime-linux-aarch64-${RKNPU2_VERSION}-${RKNN2_TARGET_SOC}.tgz")
set(RKNPU2_URL "${RKNPU2_URL_BASE}${RKNPU2_FILE}")

# download_and_decompress
download_and_decompress(${RKNPU2_URL} ${CMAKE_CURRENT_BINARY_DIR}/${RKNPU2_FILE} ${THIRD_PARTY_PATH}/install/)

# set path
set(RKNPU_RUNTIME_PATH ${THIRD_PARTY_PATH}/install/rknpu2_runtime)

# include lib
if (EXISTS ${RKNPU_RUNTIME_PATH})
    set(RKNN_RT_LIB ${RKNPU_RUNTIME_PATH}/lib/librknnrt.so)
    include_directories(${RKNPU_RUNTIME_PATH}/include)
else ()
    message(FATAL_ERROR "[rknpu2.cmake] RKNPU_RUNTIME_PATH does not exist.")
endif ()
