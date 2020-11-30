# 树莓派
PaddleX支持通过Paddle-Lite和基于OpenVINO的神经计算棒(NCS2)这两种方式在树莓派上完成预测部署。


## 硬件环境配置  

对于尚未安装系统的树莓派首先需要进行系统安装、环境配置等步骤来初始化硬件环境，过程中需要的软硬件如下：

- 硬件：micro SD，显示器，键盘，鼠标
- 软件：Raspbian OS
### Step1：系统安装
- 格式化micro SD卡为FAT格式，Windows和Mac下建议使用[SD Memory Card Formatter](https://www.sdcard.org/downloads/formatter/)工具，Linux下请参考[NOOBS For Raspberry Pi](http://qdosmsq.dunbar-it.co.uk/blog/2013/06/noobs-for-raspberry-pi/)  
- 下载NOOBS版本的Raspbian OS [下载地址](https://www.raspberrypi.org/downloads/)并将解压后的文件复制到SD中，插入SD后给树莓派通电，然后将自动安装系统
### Step2：环境配置
- 启用VNC和SSH服务：打开LX终端输入，输入如下命令，选择Interfacing Option然后选择P2 SSH 和 P3 VNC分别打开SSH与VNC。打开后就可以通过SSH或者VNC的方式连接树莓派
```
sudo raspi-config
```
- 更换源：由于树莓派官方源速度很慢，建议在官网查询国内源 [树莓派软件源](https://www.jianshu.com/p/67b9e6ebf8a0)。更换后执行
```
sudo apt-get update
sudo apt-get upgrade
```

## Paddle-Lite部署
基于Paddle-Lite的部署目前可以支持PaddleX的分类、分割与检测模型，其中检测模型仅支持YOLOV3  

部署的流程包括：PaddleX模型转换与转换后的模型部署  

**说明**：PaddleX安装请参考[PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html)，Paddle-Lite详细资料请参考[Paddle-Lite](https://paddle-lite.readthedocs.io/zh/latest/index.html)

请确保系统已经安装好上述基本软件，并配置好相应环境，**下面所有示例以工作目录 `/root/projects/`演示**。

### Paddle-Lite模型转换
将PaddleX模型转换为Paddle-Lite模型，具体请参考[Paddle-Lite模型转换](./export_nb_model.md)

### Paddle-Lite 预测
#### Step1 下载PaddleX预测代码
```
mkdir -p /root/projects
cd /root/projects
git clone https://github.com/PaddlePaddle/PaddleX.git
```
**说明**：其中C++预测代码在PaddleX/deploy/raspberry 目录，该目录不依赖任何PaddleX下其他目录，如果需要在python下预测部署请参考[Python预测部署](./python.md)。  

#### Step2：Paddle-Lite预编译库下载
对于Armv7hf的用户提供了2.6.1版本的Paddle-Lite在架构为armv7hf的ArmLinux下面的Full版本预编译库:[Paddle-Lite(ArmLinux)预编译库](https://bj.bcebos.com/paddlex/deploy/lite/inference_lite_2.6.1_armlinux.tar.bz2)    
对于Armv8的用户提供了2.6.3版本的Paddle-Lite在架构为armv8的ArmLinux下面的full版本预编译库:[Paddle-Lite(ArmLinux)与编译库](https://bj.bcebos.com/paddlex/paddle-lite/armlinux/paddle-Lite_armlinux_full_2.6.3.zip)  
其他版本与arm架构的Paddle-Lite预测库请在官网[Releases](https://github.com/PaddlePaddle/Paddle-Lite/release)下载
若用户需要在树莓派上自行编译Paddle-Lite，在树莓派上LX终端输入  

```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
sudo ./lite/tools/build.sh  --arm_os=armlinux --arm_abi=armv7hf --arm_lang=gcc  --build_extra=ON full_publish
```
预编库位置：`./build.lite.armlinux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx`  

**注意**：预测库版本需要跟opt版本一致，检测与分割请使用Paddle-LITE的full版本预测库,更多Paddle-Lite编译内容请参考[Paddle-Lite编译](https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html)；更多预编译Paddle-Lite预测库请参考[Paddle-Lite Release Note](https://github.com/PaddlePaddle/Paddle-Lite/releases)

#### Step3 软件依赖
提供了依赖软件的预编包或者一键编译，对于armv7树莓派用户不需要单独下载或编译第三方依赖软件，对于armv8树莓派用户需要自行编译opencv，编译第三方依赖软件请参考：

- gflags：编译请参考 [编译文档](https://gflags.github.io/gflags/#download)  

- opencv: 编译请参考
[编译文档](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)  
**注意**:对于armv8树莓派用户，需要自行编译opencv  

#### Step4: 编译
编译`cmake`的命令在`scripts/build.sh`中，修改LITE_DIR为Paddle-Lite预测库目录，若自行编译第三方依赖软件请根据Step1中编译软件的实际情况修改主要参数，其主要内容说明如下：
```
# Paddle-Lite预编译库的路径
LITE_DIR=/path/to/Paddle-Lite/inference/lib
# gflags预编译库的路径，若没有自行编译无需修改
GFLAGS_DIR=$(pwd)/deps/gflags
# opencv预编译库的路径，若自行编译请指定到对应路径
OPENCV_DIR=$(pwd)/deps/opencv/
# arm处理器架构 armv7或者armv8
ARCH=armv7
# Lite预测库版本 light或者full
LITE=full
```
执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```  


#### Step5: 预测

编译成功后，分类任务的预测可执行程序为`classifier`,分割任务的预测可执行程序为`segmenter`，检测任务的预测可执行程序为`detector`，其主要命令参数说明如下：  

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 模型转换生成的.xml文件路径，请保证模型转换生成的三个文件在同一路径下|
| --image  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --thread_num | 预测的线程数，默认值为1 |
| --cfg_file | PaddleX model 的.yml配置文件 |
| --save_dir | 可视化结果图片保存地址，仅适用于检测和分割任务，默认值为" "既不保存可视化结果 |

#### 样例
`样例一`：
单张图片分类任务  
测试图片 `/path/to/test_img.jpeg`  

```shell
./build/classifier --model_dir=/path/to/nb_model
--image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml  --thread_num=4
```


`样例二`:
多张图片分割任务
预测多个图片`/path/to/image_list.txt`，image_list.txt内容的格式如下：
```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```shell
./build/segmenter --model_dir=/path/to/models/nb_model --image_list=/root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --save_dir ./output --thread_num=4  
```  

## 性能测试
### 测试环境：
硬件：Raspberry Pi 3 Model B
系统：raspbian OS
软件：paddle-lite 2.6.1
### 测试结果
单位ms，num表示paddle-lite下使用的线程数  

|模型|lite(num=4)|输入图片大小|
| ----|  ---- | ----|
|mobilenet-v2|136.19|224*224|
|resnet-50|1131.42|224*224|
|deeplabv3|2162.03|512*512|
|hrnet|6118.23|512*512|
|yolov3-darknet53|4741.15|320*320|
|yolov3-mobilenet|1424.01|320*320|
|densenet121|1144.92|224*224|
|densenet161|2751.57|224*224|
|densenet201|1847.06|224*224|
|HRNet_W18|1753.06|224*224|
|MobileNetV1|177.63|224*224|
|MobileNetV3_large_ssld|133.99|224*224|
|MobileNetV3_small_ssld|53.99|224*224|
|ResNet101|2290.56|224*224|
|ResNet101_vd|2337.51|224*224|
|ResNet101_vd_ssld|3124.49|224*224|
|ShuffleNetV2|115.97|224*224|
|Xception41|1418.29|224*224|
|Xception65|2094.7|224*224|  


从测试结果看建议用户在树莓派上使用MobileNetV1-V3,ShuffleNetV2这类型的小型网络

## NCS2部署
树莓派支持通过OpenVINO在NCS2上跑PaddleX模型预测，目前仅支持PaddleX的分类网络，基于NCS2的方式包含Paddle模型转OpenVINO IR以及部署IR在NCS2上进行预测两个步骤。
- 模型转换请参考：[PaddleX模型转换为OpenVINO IR](../openvino/export_openvino_model.md)，raspbian OS上的OpenVINO不支持模型转换，需要先在host侧转换FP16的IR。
- 预测部署请参考[OpenVINO部署](../openvino/linux.md)中VPU在raspbian OS部署的部分  
- 目前仅支持armv7的树莓派
