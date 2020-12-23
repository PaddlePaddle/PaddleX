# Paddle-Lite预编译库的路径
LITE_DIR=/path/to/Paddle-Lite/inference/lib

# gflags预编译库的路径
GFLAGS_DIR=$(pwd)/deps/gflags

# opencv预编译库的路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv

# arm处理器架构，默认为armv7下ARCH=armv7-a，若CPU为armv8请修改为ARCH=armv8-a
ARCH=armv7-a

# 采用lite的版本，可选为full 与 light，默认为full版
LITE=full

# 下载自带预编译版本
bash $(pwd)/scripts/install_third-party.sh

rm -rf build
mkdir -p build
cd build
cmake .. \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR} \
    -DLITE_DIR=${LITE_DIR} \
    -DARCH=${ARCH} \
    -DLITE=${LITE} \
    -DCMAKE_CXX_FLAGS="-march=${ARCH}"
make -j4
