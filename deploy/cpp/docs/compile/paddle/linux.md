# 基于PaddleInference的推理-Linux环境编译

本文档指引用户如何基于PaddleInference对飞桨模型进行推理，并编译执行。

## 环境依赖
gcc >= 5.4.0
cmake >= 3.5.1

Ubuntu 16.04/18.04 ([我的Linux系统不在这里怎么办？]())

## 部署支持
目前部署支持的模型包括
1. PaddleX 训练导出的模型
2. PaddleDetection release-0.5版本导出的模型（仅支持FasterRCNN/MaskRCNN/PPYOLO/YOLOv3)
3. PaddleSeg release-2.0版本导出的模型
4. PaddleClas release-2.0版本导出的模型

## 编译示例Demo
### Step 1. 获取代码
```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout deploykit
cd deploy/cpp
```
在`deploy/cpp`目录中，所有的公共实现代码在`model_deploy`目录下，而示例demo代码为`demo/model_infer.cpp`。

### Step 2. 下载PaddleInference推理库
当前支持高于2.0版本以上的PaddleInference推理库，用户可根据需求在[官网页面](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#linux)下载。
下载后解压至`PaddleX/deploy/cpp`目录下

### Step 3. 修改编译参数
根据自己的系统环境，修改`PaddleX/deploy/cpp/script/build.sh`脚本中的参数，主要修改的参数为以下几个
| 参数 | 说明 |
| :--- | :--- |
| WITH_GPU | ON或OFF，表示是否使用GPU，当下载的为CPU预测库时，设为OFF |
| PADDLE_DIR | 预测库所在路径，默认为`PaddleX/deploy/cpp/paddle_inference`目录下 |
| CUDA_LIB | cuda相关lib文件所在的目录路径 |
| CUDNN_LIB | cudnn相关lib文件所在的目录路径 |

### Step 4. 编译
**[注意]**: 以下命令在`PaddleX/deploy/cpp`目录下进行执行
```
sh script/build.sh
```
在编译过程中，会调用`script/bootstrap.sh`联网下载opencv的依赖，以及yaml的依赖
