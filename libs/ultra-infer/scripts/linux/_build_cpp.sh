#!/bin/bash

set -e

# deal cmd input
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --with-gpu) WITH_GPU="$2"; shift ;;
        --enable-benchmark) ENABLE_BENCHMARK="$2"; shift ;;
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
export CC=/usr/local/gcc-8.2/bin/gcc
export CXX=/usr/local/gcc-8.2/bin/g++

cd /workspace/ultra-infer

wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.cloud.tencent.com/repo/centos7_base.repo
yum clean all
yum makecache

yum install -y patchelf rapidjson-devel

ln -sf /opt/_internal/cpython-3.10.0/bin/python3.10 /usr/bin/python
ln -sf /opt/_internal/cpython-3.10.0/bin/pip3.10 /usr/bin/pip

export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib:${LD_LIBRARY_PATH}
export PATH=/opt/_internal/cpython-3.10.0/bin:${PATH}

if [ "$ENABLE_TRT_BACKEND" = "ON" ] && [ "$TRT_DIRECTORY" = "Default" ]; then
    TRT_VERSION='8.5.2.2'
    CUDA_VERSION='11.8'
    CUDNN_VERSION='8.6'
    rm -rf "TensorRT-${TRT_VERSION}" "TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.cudnn${CUDNN_VERSION}.tar.gz"
    http_proxy= https_proxy= wget "https://fastdeploy.bj.bcebos.com/resource/TensorRT/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.cudnn${CUDNN_VERSION}.tar.gz"
    tar -xzvf "TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.cudnn${CUDNN_VERSION}.tar.gz"
    TRT_DIRECTORY="/workspace/ultra-infer/TensorRT-${TRT_VERSION}"
fi

(
    cd /workspace/ultra-infer
    rm -rf build && mkdir build && cd build
    unset http_proxy https_proxy
    cmake \
        -DLIBRARY_NAME='ultra_infer_runtime' \
        -DCMAKE_INSTALL_PREFIX="${PWD}/ultra_infer_install" \
        -DWITH_GPU="${WITH_GPU}" \
        -DENABLE_TRT_BACKEND="${ENABLE_TRT_BACKEND}"  \
        -DTRT_DIRECTORY="${TRT_DIRECTORY}"  \
        -DENABLE_ORT_BACKEND="${ENABLE_ORT_BACKEND}"  \
        -DENABLE_PADDLE_BACKEND="${ENABLE_PADDLE_BACKEND}"  \
        -DPADDLEINFERENCE_URL="${PADDLEINFERENCE_URL}" \
        -DPADDLEINFERENCE_VERSION="${PADDLEINFERENCE_VERSION}" \
        -DENABLE_OPENVINO_BACKEND="${ENABLE_OPENVINO_BACKEND}" \
        -DENABLE_VISION="${ENABLE_VISION}" \
        -DENABLE_TEXT="${ENABLE_TEXT}" \
        -DBUILD_ULTRAINFER_PYTHON=OFF \
        -DBUILD_FD_TRITON_BACKEND=ON \
        -DENABLE_BENCHMARK="${ENABLE_BENCHMARK}" \
        ..
    make -j"$(nproc)"
    make install
)
