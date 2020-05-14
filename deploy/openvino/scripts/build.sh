# openvino预编译库的路径
OPENVINO_DIR=/path/to/inference_engine/
# gflags预编译库的路径
GFLAGS_DIR=/path/to/gflags
# ngraph lib的路径，编译openvino时通常会生成
NGRAPH_LIB=/path/to/ngraph/lib/

# opencv预编译库的路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/
# 下载自带预编译版本
sh $(pwd)/scripts/bootstrap.sh

rm -rf build
mkdir -p build
cd build
cmake .. \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR} \
    -DOPENVINO_DIR=${OPENVINO_DIR} \
    -DNGRAPH_LIB=${NGRAPH_LIB} 
make
