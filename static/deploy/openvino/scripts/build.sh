# OpenVINO预编译库的路径
OPENVINO_DIR=$INTEL_OPENVINO_DIR/inference_engine

# ngraph lib的路径，编译openvino时通常会生成
NGRAPH_LIB=$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib

# gflags预编译库的路径
GFLAGS_DIR=$(pwd)/deps/gflags

# opencv使用自带预编译版本
OPENCV_DIR=$(pwd)/deps/opencv/

#cpu架构
ARCH=x86
export ARCH

#下载并编译third-part lib
sh $(pwd)/scripts/install_third-party.sh

rm -rf build
mkdir -p build
cd build
if [ ${ARCH} = "x86" ];then
  cmake .. \
      -DOPENCV_DIR=${OPENCV_DIR} \
      -DGFLAGS_DIR=${GFLAGS_DIR} \
      -DOPENVINO_DIR=${OPENVINO_DIR} \
      -DNGRAPH_LIB=${NGRAPH_LIB} \
      -DARCH=${ARCH}
  make
else
  cmake ..\
      -DOPENCV_DIR=${OPENCV_DIR} \
      -DGFLAGS_DIR=${GFLAGS_DIR} \
      -DOPENVINO_DIR=${OPENVINO_DIR} \
      -DNGRAPH_LIB=${NGRAPH_LIB} \
      -DARCH=${ARCH} \
      -DCMAKE_CXX_FLAGS="-march=armv7-a"
  make
fi
