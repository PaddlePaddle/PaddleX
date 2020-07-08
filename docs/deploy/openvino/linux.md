# Linux平台
PaddleX支持将训练好的Paddle模型部署到装有openvino的硬件上，通过openvino实现对模型预测的加速


## 部署环境

* Ubuntu* 16.04 (64-bit) with GCC* 5.4.0
* CMake 3.0+
* Python 3.7+
* ONNX 1.5.0
* PaddleX 1.0+
* OpenVINO 2020.3   
  
**说明**：PaddleX安装请参考[PaddleX](https://paddlex.readthedocs.io/zh_CN/latest/install.html) ， OpenVINO安装请参考[OpenVINO-Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录 `/root/projects/`演示**。

 

## 部署流程  
**PaddleX到OpenVINO的部署流程可以分为如下两步**： 

  * **模型转换**:将paddle的模型转换为openvino的Inference Engine
  * **预测部署**:使用Inference Engine进行预测
  

## 模型转换 
**模型转换请参考文档[模型转换](./export_openvino_model.html)**
  


## 预测部署  

### 说明
文档提供了c++下预测部署的方法，如果需要在python下预测部署请参考[python预测部署](./python.html)

### Step1 软件依赖

- gflags：编译请参考 [编译文档](https://gflags.github.io/gflags/#download)  

- opencv: 编译请参考 
[编译文档](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
说明：/root/projects/openvino/scripts/bootstrap.sh提供了预编译版本下载，也可自行编译。  

  
### Step2: 编译
编译`cmake`的命令在`scripts/build.sh`中，请根据Step1中编译软件的实际情况修改主要参数，其主要内容说明如下：
```
# openvino预编译库的路径
OPENVINO_DIR=$INTEL_OPENVINO_DIR/inference_engine
# gflags预编译库的路径
GFLAGS_DIR=/path/to/gflags/build/
# ngraph lib预编译库的路径
NGRAPH_LIB=$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib
# opencv预编译库的路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/
```
修改脚本设置好主要参数后，执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```  

### Step3: 预测

编译成功后，分类任务的预测可执行程序为`classifier`,分割任务的预测可执行程序为`segmenter`，其主要命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 模型转换生成的.xml文件路径，请保证模型转换生成的三个文件在同一路径下|
| --image  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --device  | 运行的平台, 默认值为"CPU" |
| --cfg_dir | PaddleX model 的.yml配置文件 |


### 样例
`样例一`：

测试图片 `/path/to/test_img.jpeg`  

```shell
./build/classifier --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_dir=/path/to/PadlleX_model.yml
```


`样例二`:

预测多个图片`/path/to/image_list.txt`，image_list.txt内容的格式如下：
```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```shell
./build/classifier --model_dir=/path/to/models/openvino_model --image_list=/root/projects/images_list.txt --cfg_dir=/path/to/PadlleX_model.yml
```