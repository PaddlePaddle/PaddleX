# OpenVINO预编译库inference_engine的路径
OPENVINO_DIR=$INTEL_OPENVINO_DIR/inference_engine

# ngraph lib的路径，编译openvino时通常会生成
NGRAPH_LIB=$INTEL_OPENVINO_DIR/deployment_tools/ngraph

# gflags预编译库的路径
GFLAGS_DIR=$(pwd)/deps/gflags

# opencv使用自带预编译版本
OPENCV_DIR=$(pwd)/deps/opencv/

#cpu架构
ARCH=x86
export ARCH

#下载并编译 gflags
GFLAGS_URL=https://bj.bcebos.com/paddlex/deploy/gflags.tar.gz
if [ ! -d ${GFLAGS_DIR} ]; then
    cd deps
    wget -c ${GFLAGS_URL} -O glog.tar.gz
    tar -zxvf glog.tar.gz
    rm -rf glog.tar.gz
    cd ..
fi

mkdir -p deps
# opencv
if [ "$ARCH" = "x86" ]; then
    OPENCV_URL=https://bj.bcebos.com/paddlex/deploy/x86opencv/opencv.tar.bz2
else
    OPENCV_URL=https://bj.bcebos.com/paddlex/deploy/armlinux/opencv.tar.bz2
fi
if [ ! -d "./deps/opencv" ]; then
    cd deps
    wget -c ${OPENCV_URL}
    tar xvfj opencv.tar.bz2
    rm -rf opencv.tar.bz2
    cd ..
fi

rm -rf build
mkdir -p build
cd build
if [ ${ARCH} = "x86" ];then
  cmake .. \
      -DWITH_OPENVINO=ON \
      -DOPENCV_DIR=${OPENCV_DIR} \
      -DGFLAGS_DIR=${GFLAGS_DIR} \
      -DOPENVINO_DIR=${OPENVINO_DIR} \
      -DNGRAPH_LIB=${NGRAPH_LIB} \
      -DARCH=${ARCH}
  make -j16
else
  cmake .. \
      -DWITH_OPENVINO=ON \
      -DOPENCV_DIR=${OPENCV_DIR} \
      -DGFLAGS_DIR=${GFLAGS_DIR} \
      -DOPENVINO_DIR=${OPENVINO_DIR} \
      -DNGRAPH_LIB=${NGRAPH_LIB} \
      -DARCH=${ARCH} \
      -DCMAKE_CXX_FLAGS="-march=armv7-a"
  make
fi
