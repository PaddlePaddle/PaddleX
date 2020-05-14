WITH_STATIC_LIB=OFF
OPENVINO_LIB=/usr/local/deployment_tools/inference_engine/
# OPENCV 路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/
sh $(pwd)/scripts/bootstrap.sh
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DOPENVINO_LIB=${OPENVINO_LIB} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB}
make
