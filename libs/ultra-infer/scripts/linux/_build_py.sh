#!/bin/bash

set -e

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --with-gpu) WITH_GPU="$2"; shift ;;
        --enable-benchmark) ENABLE_BENCHMARK="$2"; shift ;;
        --python) PYTHON_VERSION="$2"; shift ;;
        --paddleinference-url) PADDLEINFERENCE_URL="$2"; shift ;;
        --paddleinference-version) PADDLEINFERENCE_VERSION="$2"; shift ;;
        --enable-paddle-backend) ENABLE_PADDLE_BACKEND="$2"; shift ;;
        --enable-ort-backend) ENABLE_ORT_BACKEND="$2"; shift ;;
        --enable-openvino-backend) ENABLE_OPENVINO_BACKEND="$2"; shift ;;
        --enable-trt-backend) ENABLE_TRT_BACKEND="$2"; shift ;;
        --trt-directory) TRT_DIRECTORY="$2"; shift ;;
        --enable-vision) ENABLE_VISION="$2"; shift ;;
        --enable-text) ENABLE_TEXT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export DEBIAN_FRONTEND='noninteractive'
export TZ='Asia/Shanghai'

cd /workspace

wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.cloud.tencent.com/repo/centos7_base.repo
yum clean all
yum makecache

yum --disablerepo=epel -y update ca-certificates
yum install -y wget bzip2
yum install -y epel-release
yum install -y patchelf rapidjson-devel

PYTHON_DIR="/opt/_internal/cpython-${PYTHON_VERSION}"
if [ -d "$PYTHON_DIR" ]; then
    ln -sf "${PYTHON_DIR}/bin/python${PYTHON_VERSION}" /usr/bin/python
    ln -sf "${PYTHON_DIR}/bin/pip${PYTHON_VERSION}" /usr/bin/pip
    export LD_LIBRARY_PATH="${PYTHON_DIR}/lib:${LD_LIBRARY_PATH}"
    export PATH="${PYTHON_DIR}/bin:${PATH}"
else
    echo "Python version ${PYTHON_VERSION} not found in ${PYTHON_DIR}."
    exit 1
fi

python -m pip install numpy pandas

cd /workspace/ultra-infer

if [ "$ENABLE_TRT_BACKEND" = "ON" ] && [ "$TRT_DIRECTORY" = "Default" ]; then
    TRT_VERSION='8.5.2.2'
    CUDA_VERSION='11.8'
    CUDNN_VERSION='8.6'
    rm -rf "TensorRT-${TRT_VERSION}" "TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.cudnn${CUDNN_VERSION}.tar.gz"
    http_proxy= https_proxy= wget "https://fastdeploy.bj.bcebos.com/resource/TensorRT/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.cudnn${CUDNN_VERSION}.tar.gz"
    tar -xzvf "TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.cudnn${CUDNN_VERSION}.tar.gz"
    TRT_DIRECTORY="/workspace/ultra-infer/TensorRT-${TRT_VERSION}"
fi

export WITH_GPU="${WITH_GPU}"
export ENABLE_TRT_BACKEND="${ENABLE_TRT_BACKEND}"
export TRT_DIRECTORY="${TRT_DIRECTORY}"
export ENABLE_ORT_BACKEND="${ENABLE_ORT_BACKEND}"
export ENABLE_PADDLE_BACKEND="${ENABLE_PADDLE_BACKEND}"
export PADDLEINFERENCE_URL="${PADDLEINFERENCE_URL}"
export PADDLEINFERENCE_VERSION="${PADDLEINFERENCE_VERSION}"
export ENABLE_OPENVINO_BACKEND="${ENABLE_OPENVINO_BACKEND}"
export ENABLE_VISION="${ENABLE_VISION}"
export ENABLE_TEXT="${ENABLE_TEXT}"
export ENABLE_BENCHMARK="${ENABLE_BENCHMARK}"
export CC=/usr/local/gcc-8.2/bin/gcc
export CXX=/usr/local/gcc-8.2/bin/g++

cd /workspace/ultra-infer/python
python -m pip install wheel
unset http_proxy https_proxy

rm -rf .setuptools-cmake-build build ultra_infer/libs/third_libs dist
python setup.py build
# HACK
patchelf \
    --set-rpath '$ORIGIN/libs/third_libs/onnxruntime/lib:$ORIGIN/libs/third_libs/paddle2onnx/lib:$ORIGIN/libs/third_libs/paddle_inference/paddle/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/cryptopp/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/mklml/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/glog/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/protobuf/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/utf8proc/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/xxhash/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/gflags/lib:$ORIGIN/libs/third_libs/paddle_inference/third_party/install/onednn/lib:$ORIGIN/libs/third_libs/tensorrt/lib:$ORIGIN/libs/third_libs/opencv/lib64:$ORIGIN/libs/third_libs/openvino/runtime/lib:$ORIGIN/libs/third_libs/openvino/runtime/3rdparty/omp/lib' \
    build/lib.*/ultra_infer/ultra_infer_main*.so
python setup.py bdist_wheel
