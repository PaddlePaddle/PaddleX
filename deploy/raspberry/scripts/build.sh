# Paddle-Lite预编译库的路径
LITE_DIR=/path/to/Paddle-Lite/inference/lib

# gflags预编译库的路径
GFLAGS_DIR=$(pwd)/deps/gflags
# glog预编译库的路径
GLOG_DIR=$(pwd)/deps/glog

# opencv预编译库的路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv
# 下载自带预编译版本
exec $(pwd)/scripts/install_third-party.sh

rm -rf build
mkdir -p build
cd build
cmake .. \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR} \
    -DLITE_DIR=${LITE_DIR} \
    -DCMAKE_CXX_FLAGS="-march=armv7-a"  
make
