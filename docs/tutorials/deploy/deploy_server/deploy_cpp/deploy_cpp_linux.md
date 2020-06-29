# Linux平台部署

## 说明
本文档在 `Linux`平台使用`GCC 4.8.5` 和 `GCC 4.9.4`测试过，如果需要使用更高G++版本编译使用，则需要重新编译Paddle预测库，请参考: [从源码编译Paddle预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12)。

## 前置条件
* G++ 4.8.2 ~ 4.9.4
* CUDA 9.0 / CUDA 10.0, CUDNN 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录 `/root/projects/`演示**。

### Step1: 下载代码

 `git clone https://github.com/PaddlePaddle/PaddleX.git`

**说明**：其中`C++`预测代码在`/root/projects/PaddleX/deploy/cpp` 目录，该目录不依赖任何`PaddleX`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

PaddlePaddle C++ 预测库针对不同的`CPU`，`CUDA`，以及是否支持TensorRT，提供了不同的预编译版本，目前PaddleX依赖于Paddle1.8版本，以下提供了多个不同版本的Paddle预测库:

|  版本说明   | 预测库(1.8.2版本)  |
|  ----  | ----  |
| ubuntu14.04_cpu_avx_mkl  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.2-cpu-avx-mkl/fluid_inference.tgz) |
| ubuntu14.04_cpu_avx_openblas  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.2-cpu-avx-openblas/fluid_inference.tgz) |
| ubuntu14.04_cpu_noavx_openblas  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.2-cpu-noavx-openblas/fluid_inference.tgz) |
| ubuntu14.04_cuda9.0_cudnn7_avx_mkl  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.2-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz) |
| ubuntu14.04_cuda10.0_cudnn7_avx_mkl  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.2-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz ) |
| ubuntu14.04_cuda10.1_cudnn7.6_avx_mkl_trt6  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.2-gpu-cuda10.1-cudnn7.6-avx-mkl-trt6%2Ffluid_inference.tgz) |

更多和更新的版本，请根据实际情况下载:  [C++预测库下载列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)

下载并解压后`/root/projects/fluid_inference`目录包含内容为：
```
fluid_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意:** 预编译版本除`nv-jetson-cuda10-cudnn7.5-trt5` 以外其它包都是基于`GCC 4.8.5`编译，使用高版本`GCC`可能存在 `ABI`兼容性问题，建议降级或[自行编译预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12)。


### Step4: 编译

编译`cmake`的命令在`scripts/build.sh`中，请根据实际情况修改主要参数，其主要内容说明如下：
```
# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=OFF
# 使用MKL or openblas
WITH_MKL=ON
# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF
# TensorRT 的路径，如果需要集成TensorRT，需修改为您实际安装的TensorRT路径
TENSORRT_DIR=/root/projects/TensorRT/
# Paddle 预测库路径, 请修改为您实际安装的预测库路径
PADDLE_DIR=/root/projects/fluid_inference
# Paddle 的预测库是否使用静态库来编译
# 使用TensorRT时，Paddle的预测库通常为动态库
WITH_STATIC_LIB=OFF
# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda/lib64
# CUDNN 的 lib 路径
CUDNN_LIB=/usr/local/cuda/lib64

# 是否加载加密后的模型
WITH_ENCRYPTION=ON
# 加密工具的路径, 如果使用自带预编译版本可不修改
sh $(pwd)/scripts/bootstrap.sh # 下载预编译版本的加密工具
ENCRYPTION_DIR=$(pwd)/paddlex-encryption

# OPENCV 路径, 如果使用自带预编译版本可不修改
sh $(pwd)/scripts/bootstrap.sh  # 下载预编译版本的opencv
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/

# 以下无需改动
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DWITH_ENCRYPTION=${WITH_ENCRYPTION} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DENCRYPTION_DIR=${ENCRYPTION_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR}
make
```
**注意：** linux环境下编译会自动下载OPENCV, PaddleX-Encryption和YAML，如果编译环境无法访问外网，可手动下载：

- [opencv3gcc4.8.tar.bz2](https://paddleseg.bj.bcebos.com/deploy/docker/opencv3gcc4.8.tar.bz2)
- [paddlex-encryption.zip](https://bj.bcebos.com/paddlex/tools/paddlex-encryption.zip)
- [yaml-cpp.zip](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)

opencv3gcc4.8.tar.bz2文件下载后解压，然后在script/build.sh中指定`OPENCE_DIR`为解压后的路径。

paddlex-encryption.zip文件下载后解压，然后在script/build.sh中指定`ENCRYPTION_DIR`为解压后的路径。

yaml-cpp.zip文件下载后无需解压，在cmake/yaml.cmake中将`URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip` 中的网址，改为下载文件的路径。

修改脚本设置好主要参数后，执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```

### Step5: 预测及可视化

**在加载模型前，请检查你的模型目录中文件应该包括`model.yml`、`__model__`和`__params__`三个文件。如若不满足这个条件，请参考[模型导出为Inference文档](../deploy_python.html#inference)将模型导出为部署格式。**  

> **注意：由于PaddleX代码的持续更新，版本低于1.0.0的模型（模型版本可查看model.yml文件中的version字段）暂时无法直接用于预测部署，参考[模型版本升级](../../upgrade_version.md)对模型版本进行升级。**  

编译成功后，预测demo的可执行程序分别为`build/demo/detector`，`build/demo/classifier`，`build/demo/segmenter`，用户可根据自己的模型类型选择，其主要命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| model_dir  | 导出的预测模型所在路径 |
| image  | 要预测的图片文件路径 |
| image_list  | 按行存储图片路径的.txt文件 |
| use_gpu  | 是否使用 GPU 预测, 支持值为0或1(默认值为0) |
| use_trt  | 是否使用 TensorRT 预测, 支持值为0或1(默认值为0) |
| gpu_id  | GPU 设备ID, 默认值为0 |
| save_dir | 保存可视化结果的路径, 默认值为"output"，**classfier无该参数** |
| key | 加密过程中产生的密钥信息，默认值为""表示加载的是未加密的模型 |
| batch_size | 预测的批量大小，默认为1 |
| thread_num | 预测的线程数，默认为cpu处理器个数 |

## 样例

可使用[小度熊识别模型](../deploy_python.html#inference)中导出的`inference_model`和测试图片进行预测，导出到/root/projects，模型路径为/root/projects/inference_model。

`样例一`：

不使用`GPU`测试图片 `/root/projects/images/xiaoduxiong.jpeg`  

```shell
./build/demo/detector --model_dir=/root/projects/inference_model --image=/root/projects/images/xiaoduxiong.jpeg --save_dir=output
```
图片文件`可视化预测结果`会保存在`save_dir`参数设置的目录下。


`样例二`:

使用`GPU`预测多个图片`/root/projects/image_list.txt`，image_list.txt内容的格式如下：
```
/root/projects/images/xiaoduxiong1.jpeg
/root/projects/images/xiaoduxiong2.jpeg
...
/root/projects/images/xiaoduxiongn.jpeg
```
```shell
./build/demo/detector --model_dir=/root/projects/inference_model --image_list=/root/projects/images_list.txt --use_gpu=1 --save_dir=output --batch_size=2 --thread_num=2
```
图片文件`可视化预测结果`会保存在`save_dir`参数设置的目录下。
