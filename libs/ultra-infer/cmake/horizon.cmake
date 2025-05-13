# get Horizon_URL
set(HORIZON_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")

set(HORIZON_VERSION "2.5.2")
set(HORIZON_FILE "horizon_runtime-xj3-aarch64-${HORIZON_VERSION}.tgz")
set(HORIZON_URL "${HORIZON_URL_BASE}${HORIZON_FILE}")

# download_and_decompress
download_and_decompress(${HORIZON_URL} ${CMAKE_CURRENT_BINARY_DIR}/${HORIZON_FILE} ${THIRD_PARTY_PATH}/install)
# set path
set(HORIZON_RUNTIME_PATH ${THIRD_PARTY_PATH}/install/)

set(DNN_PATH ${HORIZON_RUNTIME_PATH}/dnn/)
set(APPSDK_PATH ${HORIZON_RUNTIME_PATH}/appsdk/appuser/)

set(DNN_LIB_PATH ${DNN_PATH}/lib)
set(APPSDK_LIB_PATH ${APPSDK_PATH}/lib/hbbpu)
set(BPU_libs dnn cnn_intf hbrt_bernoulli_aarch64)

include_directories(${DNN_PATH}/include
                    ${APPSDK_PATH}/include)
link_directories(${DNN_LIB_PATH}
                ${APPSDK_PATH}/lib/hbbpu
                ${APPSDK_PATH}/lib)
