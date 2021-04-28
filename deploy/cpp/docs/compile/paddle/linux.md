# 基于PaddleInference的推理-Linux环境编译

本文档指引用户如何基于PaddleInference对飞桨模型进行推理，并编译执行。

## 环境依赖
gcc >= 5.4.0
cmake >= 3.5.1

Ubuntu 16.04/18.04 ([我的Linux系统不在这里怎么办？]())

## 编译步骤
### Step1: 下载PaddleX预测代码
```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout deploykit
cd deploy/cpp
```
**说明**：其中`C++`预测代码在`PaddleX/deploy/cpp` 目录，该目录不依赖任何`PaddleX`下其他目录。所有的公共实现代码在`model_deploy`目录下，而示例demo代码为`demo/model_infer.cpp`。

### Step 2. 下载PaddlePaddle C++ 预测库
PaddlePaddle C++ 预测库针对是否使用GPU、是否支持TensorRT、以及不同的CUDA版本提供了已经编译好的预测库，目前PaddleX支持Paddle预测库2.0+，最新2.0.2版本下载链接如下所示:

| 版本说明                                         | 预测库(2.0.2)                                                | 编译器  |
| ------------------------------------------------ | ------------------------------------------------------------ | ------- |
| ubuntu14.04_cpu_avx_mkl_gcc82                    | [paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.2-cpu-avx-mkl/paddle_inference.tgz) | gcc 8.2 |
| ubuntu14.04_cuda9.0_cudnn7_avx_mkl_gcc54         | [paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.2-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz) | gcc 5.4 |
| ubuntu14.04_cuda10.0_cudnn7_avx_mkl_gcc54        | [paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.2-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz) | gcc 5.4 |
| ubuntu14.04_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc82 | [ paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.2-gpu-cuda10.1-cudnn7-avx-mkl/paddle_inference.tgz) | gcc 8.2 |
| ubuntu14.04_cuda10.2_cudnn8_avx_mkl_trt7_gcc82   | [ paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.2-gpu-cuda10.2-cudnn8-avx-mkl/paddle_inference.tgz) | gcc 8.2 |
| ubuntu14.04_cuda11_cudnn8_avx_mkl_trt7_gcc82     | [ paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.2-gpu-cuda11-cudnn8-avx-mkl/paddle_inference.tgz) | gcc 8.2 |

请根据实际情况选择下载，如若以上版本不满足您的需求，请至[C++预测库下载列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#linux)选择符合的版本。

将预测库解压后，其所在目录（例如解压至`PaddleX/deploy/cpp/paddle_inferenc/`）下主要包含的内容有：

```
├── paddle/ # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息(里边有编译时gcc、cuda、cudnn的版本信息)
```

### Step 3. 修改编译参数
根据自己的系统环境，修改`PaddleX/deploy/cpp/script/build.sh`脚本中的参数，主要修改的参数为以下几个
| 参数 | 说明 |
| :--- | :--- |
| WITH_GPU | ON或OFF，表示是否使用GPU，当下载的为CPU预测库时，设为OFF |
| PADDLE_DIR | 预测库所在路径，默认为`PaddleX/deploy/cpp/paddle_inference`目录下 |
| CUDA_LIB | cuda相关lib文件所在的目录路径 |
| CUDNN_LIB | cudnn相关lib文件所在的目录路径 |

### Step 4. 编译
修改完build.sh后执行编译:  **[注意]**: 以下命令在`PaddleX/deploy/cpp`目录下进行执行

```
sh script/build.sh
```
在编译过程中，会调用`script/bootstrap.sh`联网下载opencv的依赖，以及yaml的依赖。如果无法联网下载，可手动下载：

 [yaml-cpp.zip](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)。YAML文件下载后无需解压，在`cmake/yaml.cmake`中将`URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip` 中的网址，改为下载文件的路径。

opencv需要根据ubuntu版本在`script/bootstrap.sh`中选择相应链接自行下载(比如[opencv3.4.6gcc4.8ffmpeg.tar.gz2](https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2))。下载后，在deploy/cpp目录下新建 deps目录，并将下载的opencv压缩包解压至deps目录中即可。

### Step5: 预测

**在加载模型前，请检查你的模型是部署格式, 应该包括模型、参数、配置三个文件.比如是`__model__`、`__params__`、`infer_cfg.yml` 也可以是`model.pdmodel`、`model.pdiparams`、`deploy.yaml` 三个文件，名字可能不相同。如若不是部署格式的模型，请参考[部署模型导出](../../export_model.md)将模型导出为部署格式。**  

* 编译成功后，图片预测demo的入口程序为`build/demo/model_infer`，用户可根据自己的需要以及模型类型，设置参数。其主要命令参数说明如下：

| 参数            | 说明                                                         |                    |
| --------------- | ------------------------------------------------------------ | ------------------ |
| model_filename  | 导出的预测模型 模型(model)的路径,例如`model.pdmodel`,`__model__` | 必填               |
| params_filename | 导出的预测模型 参数(params)的路径，例如`model.pdiparams`,`__params__` | 必填               |
| cfg_file        | 导出的预测模型 配置文件(yml)的路径，例如`deploy.yaml`,`infer_cfg.yml` | 必填               |
| model_type      | 导出预测模型所用的框架。当前支持的套件为[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)、[PaddleX](https://github.com/PaddlePaddle/PaddleX)对应的model_type分别为 det、seg、clas、paddlex | 必填               |
| image           | 要预测的图片文件路径                                         | 跟image_list二选一 |
| image_list      | 按行存储图片路径的.txt文件                                   | 跟image二选一      |
| use_gpu         | 是否使用 GPU 预测, 支持值为0或1(默认值为0)                   | 默认为 0           |
| use_mkl         | 是否使用 MKL加速CPU预测, 支持值为0或1(默认值为1)             | 默认为 1           |
| thread_num      | openmp对batch并行的线程数，默认为1                           | 默认为 1           |
| batch_size      | 预测的批量大小，默认为1                                      | 默认为 1           |
| gpu_id          | GPU 设备ID,默认为0                                           | 默认为0            |



## 推理运行样例

例如我们使用[[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)] release/0.5中导出的[yolov3_darknet](https://bj.bcebos.com/paddlex/deploy/models//yolov3_darknet.tar.gz)模型进行预测, 例如 模型解压路径为`PaddleX/deploy/cpp/yolov3_darknet`。用PaddleDetection套件导出的模型，所以model_type参数必须为det。

> 关于预测速度的说明：加载模型后前几张图片的预测速度会较慢，这是因为运行启动时涉及到内存显存初始化等步骤，通常在预测20-30张图片后模型的预测速度达到稳定。


### 样例一：(对单张图像做预测)

不使用`GPU`,测试图片为  `images/xiaoduxiong.jpeg`  

```shell
./build/demo/model_infer --model_filename=yolov3_darknet/__model__ --params_filename=yolov3_darknet/__params__ --cfg_file=yolov3_darknet/infer_cfg.yml --model_type=det --image=images/xiaoduxiong.jpeg --use_gpu=0

```

图片的结果会打印出来，如果要获取结果的值，可以参照demo/model_infer.cpp里的代码拿到model->results_


### 样例二：(对图像列表做预测)

使用`GPU`预测多个图片，batch_size为2。假设有个`images/image_list.txt`文件，image_list.txt内容的格式如下：

```
images/image1.jpeg
images/image2.jpeg
...
images/imagen.jpeg
```

```shell
./build/demo/model_infer --model_filename=yolov3_darknet/__model__ --params_filename=yolov3_darknet/__params__ --cfg_file=yolov3_darknet/infer_cfg.yml --model_type=det --image=images/xiaoduxiong.jpeg --use_gpu=1 --batch_size=2 --thread_num=2
```



## 多卡上运行

当前支持单机多卡部署，暂时不支持跨机器多卡。多卡部署必须使用`build/demo/multi_gpu_model_infer`进行预测，将每个batch的数据均摊到每张卡上进行并行加速。多卡的gpu_id设置几个GPU的id，就会在那几个GPU上进行预测,每个gpu_id之间用英文逗号隔开。注意：**单卡的`model_infer`只能设置一个GPU的id。**

```c++
// 每次4张图片均摊在第4、5张卡进行推理计算
./build/demo/multi_gpu_model_infer --model_filename=yolov3_darknet/__model__ --params_filename=yolov3_darknet/__params__ --cfg_file=yolov3_darknet/infer_cfg.yml --model_type=det --image=images/xiaoduxiong.jpeg --use_gpu=1 --batch_size=4 --thread_num=2 --gpu_id=4,5
  
// 每次4张图片均摊在第0、1、2、3张卡上并行推理计算
./build/demo/multi_gpu_model_infer --model_filename=yolov3_darknet/__model__ --params_filename=yolov3_darknet/__params__ --cfg_file=yolov3_darknet/infer_cfg.yml --model_type=det --image=images/xiaoduxiong.jpeg --use_gpu=1 --batch_size=4 --thread_num=2 --gpu_id=0,1,2,3
```



## 部署支持

目前部署支持的模型包括

1. PaddleX 训练导出的模型
2. PaddleDetection release-0.5版本导出的模型（仅支持FasterRCNN/MaskRCNN/PPYOLO/YOLOv3)
3. PaddleSeg release-2.0版本导出的模型
4. PaddleClas release-2.0版本导出的模型

编译完成后, 其他套件部署文档如下：

- [PaddleX部署指南](../../models/paddlex.md)
- [PaddleDetection部署指南](../../models/paddledetection.md)
- [PaddleSeg部署指南](../../models/paddleseg.md)
- [PaddleClas部署指南](../../models/paddleclas.md)