#  PaddelX-openvino
paddlex-openvino提供将PaddleX训练好的模型，部署到装有openvino的硬件上，通过openvino实现对模型预测的加速

 

## 部署流程  

**PaddleX到OoenVINO的部署流程如下**： 

  PaddleX -> ONNX -> OpenVINO IR -> OpenVINO Inference Engine


### 部署环境

* Ubuntu* 16.04 (64-bit) with GCC* 5.4.0
* CMkae 3.12.3
* Python 3.7
* ONNX 1.5.0
* PaddleX 1.0
* OpenVINO 2020.3   
  
**说明**：PaddleX安装请参考[PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/release-v1.0/README.md) ， OpenVINO相关请参考[OpenVINO](https://github.com/openvinotoolkit/openvino/blob/master/README.md)

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录 `/root/projects/`演示**。

### Step1 软件依赖
- OpenVINO：编译请参考 [编译文档](https://github.com/openvinotoolkit/openvino/blob/master/build-instruction.md) 

- gflags：编译请参考 [编译文档](https://gflags.github.io/gflags/#download)  

- opencv:编译请参考 
[编译文档](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
说明：/root/projects/openvino/scripts/bootstrap.sh提供了预编译版本下载，也可自行编译。  

- ngraph:
说明：openvino编译的过程中会生成ngraph的lib文件，位于{openvino根目录}/bin/intel64/Release/lib/下。  
### Step2: 编译
编译`cmake`的命令在`scripts/build.sh`中，请根据Step1中编译软件的实际情况修改主要参数，其主要内容说明如下：
```
# openvino预编译库的路径
OPENVINO_DIR=/path/to/openvino/inference_engine/
# gflags预编译库的路径
GFLAGS_DIR=/path/to/gflags/build/
# ngraph lib的路径，编译openvino时通常会生成
NGRAPH_LIB=/path/to/ngraph/lib/
# opencv预编译库的路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/
```
修改脚本设置好主要参数后，执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```  
### Step3: 模型转换

将PaddleX模型转换成ONNX模型：

```
paddlex --export_onnx --model_dir=/path/to/PaddleX_model --save_dir=/path/to/onnx_model  --fixed_input_shape [w,h]
```  
**说明** ：onnx请使用1.5.0版本否则可能会出现模型转换错误  

将生成的onnx模型转换为OpenVINO支持的格式

```
cd {openvino根目录}/model-optimizer
python mo_onnx.py --input_model /path/to/onnx_model --output_dir /path/to/openvino_model --input_shape [N,C,H,W]
```
**说明** ：模型转换好后包括.xml和.bin两个文件，更多细节请参考[Model Optimizer文档](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)  


### Step4: 预测

编译成功后，分类任务的预测可执行程序为`classifier`，其主要命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | Model Optimizer生成的.xml文件路径，请保证Model Optimizer生成的三个文件在同一路径下|
| --image  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --device  | 运行的平台, 默认值为"CPU" |
| --cfg_dir | PaddleX model 的.yml配置文件 |


#### 样例
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

