# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON
# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF
# 上一步下载的 Paddle 预测库路径
PADDLE_DIR=/root/projects/deps/fluid_inference/
# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda/lib64/
# CUDNN 的 lib 路径
CUDNN_LIB=/usr/local/cudnn/lib64/

# OPENCV 路径, 如果使用自带预编译版本可不设置
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/
sh $(pwd)/scripts/bootstrap.sh

# 以下无需改动
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR}
make
