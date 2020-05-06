# compile with cuda
WITH_GPU=OFF
# compile with tensorrt
WITH_TENSORRT=OFF
# path to paddle inference lib
PADDLE_DIR=/path/to/paddle_inference_lib/
# path to cuda lib
CUDA_LIB=/path/to/cuda_lib/
# path to cudnn lib 
CUDNN_LIB=/path/to/cudnn_lib/
# path to opencv lib
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/

sh $(pwd)/scripts/bootstrap.sh

rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_STATIC_LIB=OFF 
make
