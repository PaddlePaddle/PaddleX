# OpenVINO部署

## 方案简介
OpenVINO部署方案位于目录`PaddleX/deploy/openvino/`下，且独立于PaddleX其他模块，该方案目前支持在 **Linux** 完成编译和部署运行。

PaddleX到OpenVINO的部署流程如下：

> PaddleX --> ONNX --> OpenVINO IR --> OpenVINO Inference Engine

|目前支持OpenVINO部署的PaddleX模型|
|-----|
|ResNet18|
|ResNet34|
|ResNet50|
|ResNet101|
|ResNet50_vd|
|ResNet101_vd|
|ResNet50_vd_ssld|
|ResNet101_vd_ssld|
|DarkNet53|
|MobileNetV1|
|MobileNetV2|
|DenseNet121|
|DenseNet161|
|DenseNet201|

## 部署流程

### 说明
本文档在 `Ubuntu`使用`GCC 4.8.5` 进行了验证，如果需要使用更多G++版本和平台的OpenVino编译，请参考: [OpenVINO](https://github.com/openvinotoolkit/openvino/blob/2020/build-instruction.md)。


### 验证环境
* Ubuntu* 16.04 (64-bit) with GCC* 4.8.5
* CMake 3.12
* Python 2.7 or higher

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录 `/root/projects/`演示**。

```
 git clone https://github.com/PaddlePaddle/PaddleX.git
```

**说明**：其中`C++`预测代码在`/root/projects/PaddleX/deploy/openvino` 目录，该目录不依赖任何`PaddleX`下其他目录。

### Step1: 软件依赖

- openvino:
[编译文档](https://github.com/openvinotoolkit/openvino/blob/2020/build-instruction.md#build-steps)

- gflags:
[编译文档](https://gflags.github.io/gflags/#download)

- opencv:
[编译文档](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
说明：/root/projects/PaddleX/deploy/openvino/scripts/bootstrap.sh提供了预编译版本下载，也可自行编译。

- ngraph:
说明：openvino编译的过程中会生成ngraph的lib文件，位于{openvino根目录}/bin/intel64/Release/lib/下。

### Step2: 编译


编译`cmake`的命令在`scripts/build.sh`中，请根据Step1中编译软件的实际情况修改主要参数，其主要内容说明如下：
```
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
```

修改脚本设置好主要参数后，执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```

### Step3: 模型转换

将PaddleX模型转换成ONNX模型：

```
paddlex --export_onnx --model_dir=/path/to/xiaoduxiong_epoch_12 --save_dir=/path/to/onnx_model
```

将生成的onnx模型转换为OpenVINO支持的格式，请参考：[Model Optimizer文档](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

### Step4: 预测

编译成功后，分类任务的预测可执行程序为`classifier`，其主要命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | Model Optimizer生成的.xml文件路径，请保证Model Optimizer生成的三个文件在同一路径下|
| --image  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --device  | 运行的平台, 默认值为"CPU" |

#### 样例

`样例一`：

测试图片 `/path/to/xiaoduxiong.jpeg`  

```shell
./build/classifier --model_dir=/path/to/openvino_model --image=/path/to/xiaoduxiong.jpeg
```


`样例二`:

预测多个图片`/path/to/image_list.txt`，image_list.txt内容的格式如下：
```
/path/to/images/xiaoduxiong1.jpeg
/path/to/images/xiaoduxiong2.jpeg
...
/path/to/images/xiaoduxiongn.jpeg
```

```shell
./build/classifier --model_dir=/path/to/models/openvino_model --image_list=/root/projects/images_list.txt
```
