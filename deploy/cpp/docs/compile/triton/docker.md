# 基于Docker快速上手Triton部署

本文档将介绍如何基于Docker快速将Paddle转换的ONNX模型部署到Triton服务上。 并以ResNet50模型为例， 讲述整个流程。[点击下载模型](https://bj.bcebos.com/paddlex/deploy2/models/resnet50_onnx.tar.gz)

## 1 拉取Triton Docker镜像
拉取 Triton Docker镜像之前，首先需要安装[Docker](https://docs.docker.com/engine/install/)，如果需要使用GPU预测请安装[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)。

拉取镜像的命令：

- `<xx.yy>`指的是你需要拉取的Triton docker版本，目前支持`20.11`，所以请手动替换`<xx.yy>`为`20.11`。
- 镜像后缀为`-py3`为Triton的服务端（server），`-py3-clientsdk`为Triton的客户端（client）。

```
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk
```

## 2 部署Triton服务

### 2.1 准备模型库（Model Repository）

在启动Triton服务之前，我们需要准备模型库，主要包括模型文件，模型配置文件和标签文件等。模型库的概念和使用方法请参考[model_repository](https://github.com/triton-inference-server/server/blob/master/docs/model_repository.md)，模型配置文件涉及到需要关键的配置，更详细的使用方法可参考[model_configuration](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md)。 以PaddleClas中的ResNet50模型为例， 模型库的详细构成如下。



完整的模型库目录结构如下：

```
model_repository             # 模型库目录
-- resnet50_onnx             # 要加载的模型目录
  |-- 1                      # 模型版本号, 默认加载最高版本
  |   |-- model.onnx          # ResNet50转换后的onnx模型
  -- config.pbtxt            # 模型配置文件
-- model2                    # 如果有多个模型目录，会同时加载多个模型
  |-- 1  
  |   |-- model.onnx  
  -- config.pbtxt  
```

#### 模型配置文件

模型配置文件config.pbtxt按需填写，最基本配置如下，如需要更高级配置请参考[model_configuration](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md)。

```
name: "resnet50_onnx"                     # 模型名， 需与外层目录名一致
platform: "onnxruntime_onnx"              # 使用的推理引擎
max_batch_size : 0                        # 最大的batch数， 0为不限制
input [
  {
    name: "inputs"
    data_type: TYPE_FP32
    dims: [-1, 3, 224, 224]
  }
]
output [
  {
    name: "save_infer_model/scale_0.tmp_1"
    data_type: TYPE_FP32
    dims: [-1, 1000]
  }
]
```

**注意：** ONNX 模型可以通过设置启动参数，自动生成最基本配置文件，这样可不用在模型目录中填写config.pbtxt文件

```
启动时设置 --strict-model-config=false
# 例如
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /path/to/model_repository/:/model_repository/ nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/model_repository/ --strict-model-config=false

# 可通过curl指令获取配置文件
curl localhost:8000/v2/models/<model name>/config
# resnet50例子
curl localhost:8000/v2/models/resnet50_onnx/config
```



#### Paddle Inference模型转为ONNX模型

1.通过[PaddleClas模型部署指南](../../models/paddleclas.md) 得到Paddle Inference类型的ResNet50模型，其他套件模型请参考：[PaddleDetection模型部署指南](../../models/paddledetection.md) 、[PaddleSeg模型部署指南](../../models/paddleseg.md)

```
ResNet50
  |-- model.pdiparams        # 静态图模型参数
  |-- model.pdiparams.info   # 参数额外信息，一般无需关注
  |-- model.pdmodel          # 静态图模型文件
  |-- resnet50_imagenet.yml  # 配置文件
```

2.将paddle inference模型转为onnx模型， 详细可参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)文档

ResNet50模型转换如下，转换后模型输出在 onnx_models/resnet50_onnx/model.onnx。

```
paddle2onnx --model_dir path/to/ResNet50  --save_file onnx_models/resnet50_onnx/model.onnx  --opset_version 9 --enable_onnx_checker True --model_filename model.pdmodel --params_filename model.pdiparams
```

**注意：**

- 留意模型转换的输出，根据提示调整opset_version的值
- paddle inference模型中配置文件(如 resnet50_imagenet.yml)在进行推理请求时会用到



### 2.2 启动Triton server服务

经过Triton的优化可以使用GPU提供极佳的推理性能，且可以在仅支持CPU的系统上工作。以上两种情况下，我们都可以使用上述的Triton Docker镜像部署。

#### 2.2.1 启动基于GPU的服务

使用以下命令对刚刚创建的ResNet模型库运行Triton服务。必须安装[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)，Docker才能识别GPU。 --gpus = 1参数表明Triton可以使用1块GPU进行推理。

```
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /path/to/model_repository/:/model_repository/
 nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/model_repository/
```

启动Triton之后，您将在控制台上看到如下输出，显示服务器正在启动并加载模型。当您看到如下输出时，Triton已加载好模型，可进行推理请求。

```
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| <model_name>         | <v>     | READY  |
| ..                   | .       | ..     |
| ..                   | .       | ..     |
+----------------------+---------+--------+
...
...
...
I1002 21:58:57.891440 62 grpc_server.cc:3914] Started GRPCInferenceService at 0.0.0.0:8001
I1002 21:58:57.893177 62 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I1002 21:58:57.935518 62 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

Triton会逐个目录加载模型，如果加载不成功，则会报告失败以及失败的原因。所有模型均应显示“ READY”状态，表示已正确加载。如果您的模型未显示在表中，请检查模型库和CUDA驱动程序的路径。

#### 2.2.2 启动仅支持CPU的服务

在没有GPU的系统上，请在不使用--gpus参数的情况下运行，其他参数与上述启动GPU部署服务的命令一致。

```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /path/to/model_repository/:/model_repository/
 nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/model_repository/
```

### 2.3 验证远程服务

使用Triton的ready接口来验证服务器和模型是否已准备好进行推断

```
curl -v localhost:8000/v2/health/ready

打印：
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

## 3 推理请求

推理请求需要基于Triton的客户端代码，对部署代码进行编译。

### 3.1 获取部署代码

```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
```

### 3.2 启动Triton客户端容器

启动容器时，需要-v参数把部署代码挂载进容器

```
# 启动前需把paddle模型的配置文件移到挂载目录，ResNet50模型为例
mv ResNet50/resnet50_imagenet.yml /path/to/PaddleX/deploy/cpp
# 启动容器
docker run -it --net=host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk -v /path/to/PaddleX/:/PaddleX/ bash
```

### 3.3 部署代码编译

打开项目路径，运行编译脚本

```
cd /PaddleX/deploy/cpp/
sh scripts/triton_build.sh --triton_client=/workspace/install/
```

### 3.4 进行推理请求

编译后会在`PaddleX/deploy/cpp/build/demo`目录下生成`model_infer`可执行二进制文件示例，用于进行模型预处理、推理请求以及后处理。

以[ResNet50](https://bj.bcebos.com/paddlex/deploy2/models/resnet50_onnx.tar.gz)为例，执行下面的指令即可进行推理请求。

```
./build/demo/model_infer --image resnet50/test.jpeg  --cfg_file resnet50/infer_cfg.yml --url localhost:8000 --model_name resnet50_onnx --model_type clas
```

**参数说明**

| 参数名称      | 含义                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------- |
| model_name    | 模型名称                                                                                          |
| url           | 服务的远程IP地址+端口，如：localhost:8000                                                         |
| model_version | 模型版本号，默认为1                                                                               |
| cfg_file      | Paddle Inference模型配置文件路径，如`resnet50/infer_cfg.yml`                                      |
| image         | 需要预测的单张图片的文件路径                                                                      |
| image_list    | 待预测的图片路径列表文件路径，列表里每一行是一张图片的文件路径                                    |
| model_type    | 模型来源，det/seg/clas/paddlex，分别表示模型来源于PaddleDetection、PaddleSeg、PaddleClas和PaddleX |
