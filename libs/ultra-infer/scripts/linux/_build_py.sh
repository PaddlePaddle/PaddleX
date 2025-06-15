#!/bin/bash

set -e

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --with-gpu) WITH_GPU="$2"; shift ;;
        --enable-benchmark) ENABLE_BENCHMARK="$2"; shift ;;
        --python) PYTHON_VERSION="$2"; shift ;;
        --enable-ort-backend) ENABLE_ORT_BACKEND="$2"; shift ;;
        --enable-openvino-backend) ENABLE_OPENVINO_BACKEND="$2"; shift ;;
        --enable-trt-backend) ENABLE_TRT_BACKEND="$2"; shift ;;
        --trt-directory) TRT_DIRECTORY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export DEBIAN_FRONTEND='noninteractive'
export TZ='Asia/Shanghai'

cd /workspace

if [ "$CUDA_VERSION" = "11.8" ]; then
    wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.cloud.tencent.com/repo/centos7_base.repo
fi
yum clean all
yum makecache

yum -y update ca-certificates
yum install -y wget bzip2
yum install -y epel-release
yum install -y patchelf rapidjson-devel

PYTHON_DIR="/opt/_internal/cpython-${PYTHON_VERSION}"
if [ -d "$PYTHON_DIR" ]; then
    export LD_LIBRARY_PATH="${PYTHON_DIR}/lib:${LD_LIBRARY_PATH}"
    export PATH="${PYTHON_DIR}/bin:${PATH}"
else
    echo "Python version ${PYTHON_VERSION} not found in ${PYTHON_DIR}."
    exit 1
fi

python -m pip install numpy pandas

cd /workspace/ultra-infer

if [[ "$CUDA_VERSION" = "12.6" || "$CUDA_VERSION" = "12.9" ]]; then
    CC=/opt/rh/gcc-toolset-11/root/usr/bin/gcc
    CXX=/opt/rh/gcc-toolset-11/root/usr/bin/g++
    ENABLE_TRT_BACKEND=OFF
    TRT_DIRECTORY=""
else
    TRT_VERSION='8.6.1.6'
    CUDA_VERSION='11.8'
    CUDNN_VERSION='8.6'
    if [ "$ENABLE_TRT_BACKEND" = "ON" ] && [ "$TRT_DIRECTORY" = "Default" ]; then
        rm -rf "TensorRT-${TRT_VERSION}" "TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz"
        http_proxy= https_proxy= wget "https://fastdeploy.bj.bcebos.com/resource/TensorRT/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz"
        tar -xzvf "TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz"
        TRT_DIRECTORY="/workspace/ultra-infer/TensorRT-${TRT_VERSION}"
    fi
    CC=/usr/local/gcc-8.2/bin/gcc
    CXX=/usr/local/gcc-8.2/bin/g++
fi

export WITH_GPU="${WITH_GPU}"
export ENABLE_TRT_BACKEND="${ENABLE_TRT_BACKEND}"
export TRT_DIRECTORY="${TRT_DIRECTORY}"
export ENABLE_ORT_BACKEND="${ENABLE_ORT_BACKEND}"
export ENABLE_OPENVINO_BACKEND="${ENABLE_OPENVINO_BACKEND}"
export ENABLE_BENCHMARK="${ENABLE_BENCHMARK}"
export ENABLE_PADDLE_BACKEND=OFF
export ENABLE_VISION=OFF
export ENABLE_TEXT=OFF
export CC=$CC
export CXX=$CXX

cd /workspace/ultra-infer/python
python -m pip install wheel
unset http_proxy https_proxy

rm -rf .setuptools-cmake-build build ultra_infer/libs/third_libs dist
python setup.py build
# HACK
patchelf \
    --set-rpath '$ORIGIN/libs/third_libs/onnxruntime/lib:$ORIGIN/libs/third_libs/paddle2onnx/lib:$ORIGIN/libs/third_libs/paddle_inference/paddle/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/cryptopp/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/mklml/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/glog/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/protobuf/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/utf8proc/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/xxhash/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/gflags/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/onednn/lib:$ORIGIN/libs/third_libs/tensorrt/lib:$ORIGIN/libs/third_libs/opencv/lib64:$ORIGIN/libs/third_libs/openvino/runtime/lib:$ORIGIN/libs/third_libs/openvino/runtime/3rdparty/tbb/lib' \
    build/lib.*/ultra_infer/ultra_infer_main*.so
python setup.py bdist_wheel
