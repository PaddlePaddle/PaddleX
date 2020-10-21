# OpenVINO部署简介
PaddleX支持将训练好的Paddle模型通过OpenVINO实现模型的预测加速，OpenVINO详细资料与安装流程请参考[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html)，本文档使用OpenVINO 2020.4测试通过。

## 部署支持情况
下表提供了PaddleX在不同环境下对使用OpenVINO加速的支持情况  

|硬件平台|Linux|Windows|Raspbian OS|c++|python |分类|检测|分割|
| ----|  ---- | ---- | ----|  ---- | ---- |---- | ---- |---- |
|CPU|支持|支持|不支持|支持|支持|支持|支持|不支持|
|VPU|支持|支持|支持|支持|支持|支持|不支持|不支持|  


**注意**：其中Raspbian OS为树莓派操作系统。检测模型仅支持YOLOV3，由于OpenVINO不支持ONNX的resize-11 OP的原因，目前还不支持Paddle的分割模型

## 部署流程
**PaddleX到OpenVINO的部署流程可以分为如下两步**：

  * **模型转换**:将Paddle的模型转换为OpenVINO的Inference Engine
  * **预测部署**:使用Inference Engine进行预测

## 模型转换
**模型转换请参考文档[模型转换](./export_openvino_model.md)**  
**说明**：由于不同软硬件平台下OpenVINO模型转换方法一致，故如何转换模型后续文档中不再赘述。

## 预测部署
由于不同软硬下部署OpenVINO实现预测的方式不完全一致，具体请参考：  

**[Linux](./linux.md)**:介绍了PaddleX在操作系统为Linux或者Raspbian OS，编程语言为C++，硬件平台为
CPU或者VPU的情况下使用OpenVINO进行预测加速  

**[Windows](./windows.md)**:介绍了PaddleX在操作系统为Window，编程语言为C++，硬件平台为CPU或者VPU的情况下使用OpenVINO进行预测加速  

**[Python](./python.md)**:介绍了PaddleX在python下使用OpenVINO进行预测加速
