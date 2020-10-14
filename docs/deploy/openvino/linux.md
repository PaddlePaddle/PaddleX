# Linux平台


## 前置条件

* OS: Ubuntu、Raspbian OS
* GCC* 5.4.0
* CMake 3.0+
* PaddleX 1.0+
* OpenVINO 2020.4
* 硬件平台：CPU、VPU

**说明**：PaddleX安装请参考[PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html) ， OpenVINO安装请根据相应的系统参考[OpenVINO-Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)或者[OpenVINO-Raspbian](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html)

请确保系统已经安装好上述基本软件，并配置好相应环境，**下面所有示例以工作目录 `/root/projects/`演示**。



## 预测部署  

文档提供了c++下预测部署的方法，如果需要在python下预测部署请参考[python预测部署](./python.md)

### Step1 下载PaddleX预测代码
```
mkdir -p /root/projects
cd /root/projects
git clone https://github.com/PaddlePaddle/PaddleX.git
```
**说明**：其中C++预测代码在PaddleX/deploy/openvino 目录，该目录不依赖任何PaddleX下其他目录。

### Step2 软件依赖
提供了依赖软件预编包或者一键编译，用户不需要单独下载或编译第三方依赖软件。若需要自行编译第三方依赖软件请参考：

- gflags：编译请参考 [编译文档](https://gflags.github.io/gflags/#download)  

- opencv: 编译请参考
[编译文档](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)



### Step3: 编译
编译`cmake`的命令在`scripts/build.sh`中，若在树莓派(Raspbian OS)上编译请修改ARCH参数x86为armv7，若自行编译第三方依赖软件请根据Step1中编译软件的实际情况修改主要参数，其主要内容说明如下：
```
# openvino预编译库的路径
OPENVINO_DIR=$INTEL_OPENVINO_DIR/inference_engine
# gflags预编译库的路径
GFLAGS_DIR=$(pwd)/deps/gflags
# ngraph lib预编译库的路径
NGRAPH_LIB=$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib
# opencv预编译库的路径
OPENCV_DIR=$(pwd)/deps/opencv/
#cpu架构（x86或armv7）
ARCH=x86
```
执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```  

### Step4: 预测

编译成功后，分类任务的预测可执行程序为`classifier`，检测任务的预测可执行程序为`detector`，其主要命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 模型转换生成的.xml文件路径，请保证模型转换生成的三个文件在同一路径下|
| --image  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --device  | 运行的平台，可选项{"CPU"，"MYRIAD"}，默认值为"CPU"，如在VPU上请使用"MYRIAD"|
| --cfg_file | PaddleX model 的.yml配置文件 |
| --save_dir | 可视化结果图片保存地址，仅适用于检测任务，默认值为" "既不保存可视化结果 |

### 样例
`样例一`：
linux系统在CPU下做单张图片的分类任务预测  
测试图片 `/path/to/test_img.jpeg`  

```shell
./build/classifier --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml
```


`样例二`:
linux系统在CPU下做多张图片的检测任务预测，并保存预测可视化结果
预测的多个图片`/path/to/image_list.txt`，image_list.txt内容的格式如下：
```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```shell
./build/detector --model_dir=/path/to/models/openvino_model --image_list=/root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --save_dir ./output
```

`样例三`:  
树莓派(Raspbian OS)在VPU下做单张图片分类任务预测
测试图片 `/path/to/test_img.jpeg`  

```shell
./build/classifier --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml --device=MYRIAD
```

## 性能测试
`测试一`：  
在服务器CPU下测试了OpenVINO对PaddleX部署的加速性能：
- CPU：Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
- OpenVINO： 2020.4
- PaddleX：采用Paddle预测库(1.8)，打开mkldnn加速，打开多线程。
- 模型来自PaddleX tutorials，Batch Size均为1，耗时单位为ms/image，只计算模型运行时间，不包括数据的预处理和后处理，20张图片warmup，100张图片测试性能。

|模型| PaddleX| OpenVINO |  图片输入大小|
|---|---|---|---|
|resnet-50 | 20.56 | 16.12 | 224*224 |
|mobilenet-V2 | 5.16 | 2.31 |224*224|
|yolov3-mobilnetv1 |76.63| 46.26|608*608 |  

`测试二`:
在PC机上插入VPU架构的神经计算棒(NCS2)，通过Openvino加速。
- CPU：Intel(R) Core(TM) i5-4300U 1.90GHz
- VPU：Movidius Neural Compute Stick2
- OpenVINO： 2020.4
- 模型来自PaddleX tutorials，Batch Size均为1，耗时单位为ms/image，只计算模型运行时间，不包括数据的预处理和后处理，20张图片warmup，100张图片测试性能。  

|模型|OpenVINO|输入图片|
|---|---|---|
|mobilenetV2|24.00|224*224|
|resnet50_vd_ssld|58.53|224*224|  

`测试三`:
在树莓派3B上插入VPU架构的神经计算棒(NCS2)，通过Openvino加速。
- CPU ：ARM Cortex-A72 1.2GHz 64bit
- VPU：Movidius Neural Compute Stick2
- OpenVINO 2020.4
- 模型来自paddleX tutorials，Batch Size均为1，耗时单位为ms/image，只计算模型运行时间，不包括数据的预处理和后处理，20张图片warmup，100张图片测试性能。  

|模型|OpenVINO|输入图片大小|
|---|---|---|
|mobilenetV2|43.15|224*224|
|resnet50|82.66|224*224|  
