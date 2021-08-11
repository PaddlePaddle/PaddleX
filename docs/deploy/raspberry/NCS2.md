# 神经计算棒2代  
PaddleX支持在树莓派上插入NCS2(神经计算棒2代)通过OpenVINO部署PadlleX训练出来的分类模型  

**注意**：目前仅支持分类模型、仅支持Armv7hf的树莓派  

## 前置条件  
* OS: Raspbian OS
* PaddleX 1.0+
* OpenVINO 2020.4  

- Raspbian OS:树莓派操作操作系统下载与安装请参考[树莓派系统安装与环境配置](./Raspberry.md#硬件环境配置)  
- PaddleX: PaddleX安装请参考[PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html)  
- OpenVINO: OpenVINO的安装请参考[OpenVINO-Raspbian](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html)  

**注意**：安装完OpenVINO后需要初始化OpenVINO环境，并且需要对USB进行配置，请参考：  

```
#初始化OpenVINO环境
source /opt/intel/openvino/bin/setupvars.sh
#将初始化OpenVINO环境的规则加入到bashrc中
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
#配置USB
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
```

## 部署流程  

部署流程主要分为模型转换与转换后模型部署两个步骤，下面以MobilnetV2模型为例，介绍如何将PaddleX训练好的模型通过OpenVINO部署到插入NCS2的树莓派  
教程的示例项目训练MobilenetV2模型，请参考[PaddleX模型训练示例](https://aistudio.baidu.com/aistudio/projectdetail/439860)  

## 模型转换

模型转换指的是将PaddleX训练出来的Paddle模型转换为OpenVINO的IR，对于模型转换教程可以参考[OpenVINO模型转换](../openvino/export_openvino_model.md)  

**注意**：树莓派上面安装的OpenVINO是不带Model Optmizier模块的，不能在上面进行模型转换，请在Host下载与树莓派一直的OpenVINO版本，然后进行模型转换。  

以转换训练好的MobileNetV2为示例，请参考以下命令:
```
#安装paddlex
pip install paddlex

#下载PaddleX代码
```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout release/1.3
```
#进入模型转换脚本文件夹
cd PaddleX/deploy/openvino/python

#导出inference模型，执行命令前务必将训练好的MobileNetV2模型拷贝到当前目录，并命名为MobileNetV2
paddlex --export_inference --model_dir ./MobileNetV2 --save_dir ./MobileNetV2 --fixed_input_shape [224,224]
#完成导出后会在MobileNetV2文件夹下面出现，__model__、__params__、model.yml三个文件

#转换Paddle inference模型到OpenVINO IR
python converter.py --model_dir ./MobileNetV2 --save_dir ./MobileNetV2 --fixed_input_shape [224,224] --data_type FP16
#转换成功后会在MobileNetV2目录下面出现 paddle2onnx.xml、paddle2onnx.mapping、paddle2onnx.bin三个文件
```

## 模型部署

PaddleX支持Python和C++两种方式在树莓派上通过NCS2部署：  
- C++：C++部署教程请参考[OpenVINO_Raspberry](../openvino/linux.md)
- python：python部署教程请参考[OpenVINO_python](../openvino/python.md)  

以转换好的MobileNetV2模型为示例  

**准备工作**
```
#在树莓派上下载PaddleX代码
```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout release/1.3
```
#进入OpenVINO部署代码
cd deploy/openvino
#将MobileNetV2转好的OpenVINO IR以及测试图片拷贝到树莓派上面，并以及MobileNetV2文件夹放到OpenVINO部署的代码的目录
```

**C++部署**
```
#修改编译文件script/build.sh，将ARCH参数修改为armv7
vim script/build.sh
#编译代码
sh script/build.sh
#OpenVINO部署
./build/classfier --model_dir MobileNetV2/paddle2onnx.xml --image [测试图片路径] --device MYRIAD --cfg_file MobileNetV2/model.yml --save_dir output
```
执行成功后会在output文件夹面保存测试图片的可视化结果  

**python部署**
```
进入python部署代码目录
cd python
#python部署
python demo.py --model_dir ../MobileNetV2/paddle2onnx.xml --img [测试图片路径] --device MYRIAD --cfg_file MobileNetV2/model.yml
```
