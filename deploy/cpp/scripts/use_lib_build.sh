# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON
# 使用MKL or openblas
WITH_MKL=ON
# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF
# TensorRT 的路径，如果需要集成TensorRT，需修改为您实际安装的TensorRT路径
TENSORRT_DIR=$(pwd)/TensorRT/
# Paddle 预测库路径, 请修改为您实际安装的预测库路径
PADDLE_DEPLOY_DIR=/ssd2/heliqi/my_project/Paddle2ONNX/deploykit/cpp/build_lib/output
# Paddle 的预测库是否使用静态库来编译
# 使用TensorRT时，Paddle的预测库通常为动态库
WITH_STATIC_LIB=ON
# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda/lib64
# CUDNN 的 lib 路径
CUDNN_LIB=/usr/lib/x86_64-linux-gnu
# 是否加载加密后的模型
WITH_ENCRYPTION=OFF
# 加密工具的路径, 如果使用自带预编译版本可不修改
ENCRYPTION_DIR=$(pwd)/paddlex-encryption


# 以下无需改动
rm -rf build
mkdir -p build
cd build
cmake ../demo_use_lib \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DWITH_ENCRYPTION=${WITH_ENCRYPTION} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DEPLOY_DIR=${PADDLE_DEPLOY_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DENCRYPTION_DIR=${ENCRYPTION_DIR} 
make -j16
