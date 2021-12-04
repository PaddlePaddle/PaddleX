# 基于PaddleInference的推理-Linux环境编译

本文档指引用户如何基于PaddleInference对飞桨模型进行推理，并编译执行。

## 环境依赖
gcc >= 5.4.0
cmake >= 3.5.1

Ubuntu 16.04/18.04

## 编译步骤
### Step1: 获取部署代码
```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX/deploy/cpp
```
**说明**：`C++`预测代码在`PaddleX/deploy/cpp` 目录，该目录不依赖任何`PaddleX`下其他目录。所有的公共实现代码在`model_deploy`目录下，所有示例代码都在`demo`目录下。

### Step 2. 下载PaddlePaddle C++ 预测库
PaddlePaddle C++ 预测库针对是否使用GPU、是否支持TensorRT、以及不同的CUDA版本提供了已经编译好的预测库，目前PaddleX支持Paddle预测库2.0+，最新2.1版本下载链接如下所示:

| 版本说明                               | 预测库(2.1)                                                                                                                   | 编译器  |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------- |
| CPU版本                                | [paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddle_inference.tgz)                     | gcc 8.2 |
| GPU版本（CUDA10.1/CUDNN7.6/TensorRT6） | [ paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddle_inference.tgz) | gcc 8.2 |
| GPU版本（CUDA10.2/CUDNN8/TensorRT7）   | [ paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddle_inference.tgz) | gcc 8.2 |
| GPU版本（CUDA11/CUDNN8/TensorRT7）     | [ paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddle_inference.tgz) | gcc 8.2 |

请根据实际情况选择下载，如若以上版本不满足您的需求，请至[C++预测库下载列表](https://paddleinference.paddlepaddle.org.cn/v2.1/user_guides/download_lib.html)选择符合的版本。

将预测库解压后，其所在目录（例如解压至`PaddleX/deploy/cpp/paddle_inference/`）下主要包含的内容有：

```
├── paddle/ # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息(里边有编译时gcc、cuda、cudnn的版本信息)
```

### Step 3. 修改编译参数
根据自己的系统环境，修改`PaddleX/deploy/cpp/script/build.sh`脚本中的参数，主要修改的参数为以下几个
| 参数          | 说明                                                                                 |
| :------------ | :----------------------------------------------------------------------------------- |
| WITH_GPU      | ON或OFF，表示是否使用GPU，当下载的为CPU预测库时，设为OFF                             |
| PADDLE_DIR    | 预测库所在路径，默认为`PaddleX/deploy/cpp/paddle_inference`目录下                    |
| CUDA_LIB      | cuda相关lib文件所在的目录路径                                                        |
| CUDNN_LIB     | cudnn相关lib文件所在的目录路径                                                       |
| WITH_TENSORRT | ON或OFF，表示是否使用开启TensorRT                                                    |
| TENSORRT_DIR  | TensorRT 的路径，如果开启TensorRT开关WITH_TENSORRT，需修改为您实际安装的TensorRT路径     |
| WITH_ENCRYPTION      | ON或OFF，表示是否开启加密模块                             |
| OPENSSL_DIR    | OPENSSL所在路径，解密所需。默认为`PaddleX/deploy/cpp/deps/penssl-1.1.0k`目录下        |

### Step 4. 编译
修改完build.sh后执行编译:  
```commandline
sh script/build.sh
```

**[注意]**:
 - 以上命令在`PaddleX/deploy/cpp`目录下进行执行
 - opencv依赖libopenblas-dev，如未安装会导致编译失败，可执行`apt-get install libopenblas-dev`进行安装（以ubuntu为例）

#### 编译环境无法联网导致编译失败？

> 编译过程，会调用script/bootstrap.sh联网下载opencv、openssl，以及yaml依赖包，如无法联网，用户按照下操作手动下载
>
> 1. 根据系统版本，点击右侧链接下载不同版本的opencv依赖 [Ubuntu 16.04](https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2)/[Ubuntu 18.04](https://bj.bcebos.com/paddlex/deploy/opencv3.4.6gcc4.8ffmpeg_ubuntu_18.04.tar.gz2)
> 2. 解压下载的opencv依赖（解压后目录名为opencv3.4.6gcc4.8ffmpeg)，创建目录`PaddleX/deploy/cpp/deps`，将解压后的目录拷贝至该创建的目录下
> 3. 点击[下载yaml依赖包](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)，无需解压
> 4. 修改`PaddleX/deploy/cpp/cmake/yaml.cmake`文件，将`URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip`中网址替换为第3步中下载的路径，如改为`URL /Users/Download/yaml-cpp.zip`
> 5. 如果**开启加密**，点击[下载openssl](https://bj.bcebos.com/paddlex/tools/openssl-1.1.0k.tar.gz),将解压后的目录拷贝至跟opencv同级目录，即`PaddleX/deploy/cpp/deps`目录。
> 6. 重新执行`sh script/build.sh`即可编译



### Step 5. 编译结果

编译后会在`PaddleX/deploy/cpp/build/demo`目录下生成`model_infer`、`multi_gpu_model_infer`和`batch_infer`等几个可执行二进制文件示例，分别用于在单卡/多卡/多batch上加载模型进行预测，示例使用参考如下文档：

- [单卡加载模型预测示例](../../demo/model_infer.md)
- [多卡加载模型预测示例](../../demo/multi_gpu_model_infer.md)

如果编译时开启TensorRT， 会多成一个`tensorrt_infer`二进制文件示例。示例使用参考如下文档：
- [PaddleInference集成TensorRT加载模型预测示例](../../demo/tensorrt_infer.md)

如果编译时开启加密， 会多成一个`decrypt_infer`二进制文件示例。示例使用参考如下文档：
- [模型加密预测示例](../../demo/decrypt_infer.md)

## 其它文档

- [PaddleClas模型部署指南](../../models/paddleclas.md)
- [PaddleDetection模型部署指南](../../models/paddledetection.md)
- [PaddleSeg模型部署指南](../../models/paddleseg.md)
