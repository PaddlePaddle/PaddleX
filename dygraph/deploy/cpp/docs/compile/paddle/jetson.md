# 基于PaddleInference的推理-Jetson环境编译

本文档指引用户如何基于Nvidia Jetpack 4.4，对飞桨模型进行推理，并编译执行。

## 编译步骤
### Step1: 获取部署代码
```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX/dygraph/deploy/cpp
```
**说明**：`C++`预测代码在`PaddleX/dygraph/deploy/cpp` 目录，该目录不依赖任何`PaddleX`下其他目录。所有的公共实现代码在`model_deploy`目录下，所有示例代码都在`demo`目录下。

### Step 2. 下载PaddlePaddle C++ 预测库
PaddlePaddle C++ 预测库针对是否使用GPU、是否支持TensorRT、以及不同的CUDA版本提供了已经编译好的预测库，目前PaddleX支持Paddle预测库2.0+，最新2.1版本下载链接如下所示:

| 版本说明                                       | 预测库(2.1)                                                                                                            |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| nv_jetson_cuda10.2_cudnn8_trt7_all(jetpack4.4) | [paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-nv-jetson-jetpack4.4-all/paddle_inference.tgz) |

**注意：**目前2.1版本的预测库只提供Nvidia Jetpack 4.4版本下载。如果你的设备是其他版本jetpack，需要重新编译Paddle预测库，请参考: [预测库源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)中的`NVIDIA Jetson嵌入式硬件预测库源码编译`。

将预测库解压后，其所在目录（例如解压至`PaddleX/dygraph/deploy/cpp/paddle_inferenc/`）下主要包含的内容有：

```
├── paddle/ # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息(里边有编译时gcc、cuda、cudnn的版本信息)
```

### Step 3. 修改编译参数
根据自己的系统环境，修改`PaddleX/dygraph/deploy/cpp/script/build.sh`脚本中的参数，主要修改的参数为以下几个
| 参数          | 说明                                                                                       |
| :------------ | :----------------------------------------------------------------------------------------- |
| WITH_GPU      | ON或OFF，表示是否使用GPU，jetson一般要开启                                                 |
| PADDLE_DIR    | 预测库所在路径，默认为`PaddleX/dygraph/deploy/cpp/paddle_inference`目录下                  |
| CUDA_LIB      | cuda相关lib文件所在的目录路径                                                              |
| CUDNN_LIB     | cudnn相关lib文件所在的目录路径                                                             |
| WITH_TENSORRT | ON或OFF，表示是否使用开启TensorRT                                                          |
| TENSORRT_DIR  | TensorRT路径，可不填。jetson编译在CMakeLists.txt中直接指定在/usr/lib/aarch64-linux-gnu目录 |

### Step 4. 编译
修改完build.sh后执行编译， **[注意]**: 以下命令在`PaddleX/dygraph/deploy/cpp`目录下进行执行

```
sh script/build.sh
```
#### 编译环境无法联网导致编译失败？

> 编译过程，会联网下载yaml依赖包，如无法联网，用户按照下操作手动下载
>
> 1. [点击下载yaml依赖包](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)，无需解压
> 2. 修改`PaddleX/deploy/cpp/cmake/yaml.cmake`文件，将`URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip`中网址替换为第3步中下载的路径，如改为`URL /Users/Download/yaml-cpp.zip`
> 3. 重新执行`sh script/build.sh`即可编译

### Step 5. 编译结果

编译后会在`PaddleX/dygraph/deploy/cpp/build/demo`目录下生成`model_infer`、`multi_gpu_model_infer`和`batch_infer`等几个可执行二进制文件示例，分别用于在单卡/多卡/多batch上加载模型进行预测，示例使用参考如下文档

- [单卡加载模型预测示例](../../demo/model_infer.md)
- [多卡加载模型预测示例](../../demo/multi_gpu_model_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](../../demo/tensorrt_infer.md)

## 其它文档

- [PaddleClas模型部署指南](../../models/paddleclas.md)
- [PaddleDetection模型部署指南](../../models/paddledetection.md)
- [PaddleSeg模型部署指南](../../models/paddleseg.md)
