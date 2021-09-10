# 基于Docker快速上手TensorRT部署

本文档将介绍如何基于Docker快速使用TensorRT部署Paddle转换的ONNX模型。并以ResNet50模型为例，讲述整个流程。[点击下载模型](https://bj.bcebos.com/paddlex/deploy2/models/resnet50_trt.tar.gz)

## 1 配置TensorRT Docker环境
拉取TensorRT Docker镜像之前，首先需要安装[Docker](https://docs.docker.com/engine/install/)，如果需要使用GPU预测请安装[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)。


拉取镜像的命令：

- `<xx.yy>`指的是你需要拉取的Tensorrt 镜像版本，以`20.11`为例，手动替换`<xx.yy>`为`20.11`：

```
$ docker pull  nvcr.io/nvidia/tensorrt:<xx.yy>-py3
```

创建一个名为 `tesnorrt-onnx` 的Docker容器：

```
$ docker run -it --gpus=all --name tensorrt-onnx  -v ~/paddle2onnx/:/paddle2onnx/ --net=host nvcr.io/nvidia/tensorrt:20.11-py3 /bin/bash
```
## 2 项目编译

拉取项目代码、TensorRT代码(依赖头文件)。进入项目路径，运行编译脚本

```
$ git clone https://github.com/PaddlePaddle/PaddleX.git
$ cd PaddleX
$ git checkout release/2.0.0
$ cd deploy/cpp
$ git clone https://github.com/NVIDIA/TensorRT.git
# 如果不是其他版本的容器， 将cuda_dir路径换成自己的cuda路径即可
$ sh scripts/tensorrt_build.sh --tensorrt_dir=/usr/lib/x86_64-linux-gnu/ --cuda_dir=/usr/local/cuda-11.1/targets/x86_64-linux/ --tensorrt_header=./TensorRT/
```

## 3 准备模型

以[ResNet50](https://bj.bcebos.com/paddlex/deploy2/models/resnet50_trt.tar.gz)为例：

### 3.1 导出Paddle Inference模型

通过[PaddleClas模型部署指南](../../models/paddleclas.md) 得到Paddle Inference类型的ResNet50模型，其他套件模型请参考：[PaddleDetection模型部署指南](../../models/paddledetection.md) 、[PaddleSeg模型部署指南](../../models/paddleseg.md)

```
ResNet50
  |-- model.pdiparams        # 静态图模型参数
  |-- model.pdiparams.info   # 参数额外信息，一般无需关注
  |-- model.pdmodel          # 静态图模型文件
  |-- resnet50_imagenet.yml  # 配置文件
```

### 3.2 转换为ONNX模型

将paddle inference模型转为onnx模型， 详细可参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)文档

ResNet50模型转换如下，转换后模型输出在 onnx_models/resnet50_onnx/model.onnx。

```
paddle2onnx --model_dir path/to/ResNet50  --save_file onnx_models/resnet50_onnx/model.onnx  --opset_version 9 --enable_onnx_checker True --model_filename model.pdmodel --params_filename model.pdiparams
```

**注意：**

- 留意模型转换的输出，根据提示调整opset_version的值
- paddle inference模型中配置文件(如 `resnet50_imagenet.yml`)包含了前后处理、标签等信息，对转换后的onnx模型进行推理时还会用到。

### 3.3 修改配置文件

在进行推理前必须在配置文件中添加确定的输入输出信息(包括名字、类型、形状)。

以上述ResNet50模型为例，在`resnet50_imagenet.yml`中加入输入输出信息(`input` `output`字段)后为：

```
model_format: Paddle
toolkit: PaddleClas
transforms:
  BGR2RGB:
    "null": true
  ResizeByShort:
    target_size: 256
    interp: 1
    use_scale: false
  CenterCrop:
    width: 224
    height: 224
  Convert:
    dtype: float
input:
  - name: "inputs"
    data_type: TYPE_FP32
    dims:
      - 1
      - 3
      - 224
      - 224
output:
  - name: "save_infer_model/scale_0.tmp_1"
    data_type: TYPE_FP32
    dims:
      - 1
      - 1000
labels:
  - kit_fox
  - English_setter
  - Siberian_husky
```

当前TensorRT部署只支持固定的输入输出，不支持动态形状(shape)。如果转换后的onnx模型的输入是动态输入，需要在配置文件的预处理transforms中加入Resize操作，将所有不同形状的图片转换为固定的形状。

例如[PaddleSeg模型部署指南](../../models/paddleseg.md) 中导出的DeepLabv3p模型，转换为onnx后是形状为[-1, 3, -1, -1]的动态输入。修改配置文件如下，可将输入固化成[1, 3, 1024, 2048]形状:

```
Deploy:
  model: model.pdmodel
  params: model.pdiparams
  transforms:
  - type: Normalize
  - type: Resize
    target_size:
      - 2048
      - 1024
input:
  - name: "x"
    data_type: TYPE_FP32
    dims:
      - 1
      - 3
      - 1024
      - 2048
output:
- name: "save_infer_model/scale_0.tmp_1"
  data_type: TYPE_FP32
  dims:
    - 1
    - 19
    - 1024
    - 2048
```

## 4 推理

上述编译后会在`PaddleX/deploy/cpp/build/demo`目录下生成`model_infer`可执行二进制文件， 用于模型预测。以[ResNet50](https://bj.bcebos.com/paddlex/deploy2/models/resnet50_trt.tar.gz)为例，执行下面的指令进行预测：

```
./build/demo/model_infer  --image resnet50/test.jpeg --cfg_file resnet50/deploy.yml --model_type clas --model_file resnet50/model.onnx
```

输出如下，结果为: Classify(类别id、标签、置信度)

```
init ClasModel,model_type=clas
start model init
start engine init
----------------------------------------------------------------
Input filename:   resnet50/model.onnx
ONNX IR version:  0.0.7
Opset version:    9
Producer name:    PaddlePaddle
Producer version:
Domain:
Model version:    0
Doc string:
----------------------------------------------------------------
WARNING: Logging before InitGoogleLogging() is written to STDERR
start model predict 1
Result for sample 0
Classify(65	Saluki	0.91879153)
```

**参数说明**

| 参数名称   | 含义                                                         |
| ---------- | ------------------------------------------------------------ |
| model_file | onnx模型路径                                                 |
| cfg_file   | Paddle Inference模型配置文件路径，如`resnet50/deploy.yml`    |
| image      | 需要预测的单张图片的文件路径                                 |
| image_list | 待预测的图片路径列表文件路径，列表里每一行是一张图片的文件路径 |
| model_type | 模型来源，det/seg/clas/paddlex，分别表示模型来源于PaddleDetection、PaddleSeg、PaddleClas和PaddleX |
| gpu_id     | 使用GPU预测时的设备ID，默认为0                               |
