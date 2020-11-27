# Paddle-Lite预编译库的路径
LITE_DIR=/home/pi/wsy/Paddle-Lite/build.lite.armlinux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx

# gflags预编译库的路径
GFLAGS_DIR=$(pwd)/deps/gflags

# opencv预编译库的路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv

# arm处理器架构，默认为armv7若CPU为armv8请修改为ARCH=armv8
ARCH=armv7

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
    -DLITE=${LITE}
make -j4
