# 基于OpenVINO的推理-Linux环境编译

本文档指引用户如何基于OpenVINO对飞桨模型进行推理，并编译执行。进行以下编译操作前请先安装好OpenVINO，OpenVINO安装请参考官网[OpenVINO-Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)

**注意：** 

- 我们测试的openvino版本为2021.3，如果你使用其它版本遇到问题，可以尝试切换到该版本
- 当前检测模型转换为openvino格式是有问题的，暂时只支持分割和分类模型

## 1 准备模型

以[ResNet50](https://bj.bcebos.com/paddlex/deploy2/models/resnet50_trt.tar.gz)为例：

### 1.1 导出Paddle Inference模型

通过[PaddleClas模型部署指南](../../models/paddleclas.md) 得到Paddle Inference类型的ResNet50模型，其他套件模型请参考：[PaddleDetection模型部署指南](../../models/paddledetection.md) 、[PaddleSeg模型部署指南](../../models/paddleseg.md)

下载的ResNet50解压后的目录结构如下：

```
ResNet50
  |-- model.pdiparams        # 静态图模型参数
  |-- model.pdiparams.info   # 参数额外信息，一般无需关注
  |-- model.pdmodel          # 静态图模型文件
  |-- resnet50_imagenet.yml  # 配置文件
```

### 1.2 转换为ONNX模型

将paddle inference模型转为onnx模型， 详细可参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)文档

ResNet50模型转换如下，转换后模型输出在 onnx_models/resnet50_onnx/model.onnx

```
# model_dir需要ResNet50解压后的路径
paddle2onnx --model_dir ResNet50  --save_file onnx_models/resnet50_onnx/model.onnx  --opset_version 9 --enable_onnx_checker True --model_filename model.pdmodel --params_filename model.pdiparams
```

### 1.3 转换为openvino模型

将onnx模型转为openvino模型， 详细可参考官网文档[转换onnx模型](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html)

以上文的ResNet50模型为例，转换指令如下:

```
# 使用mo.py 也可以
mo_onnx.py --input_model onnx_models/resnet50_onnx/model.onnx --output_dir openvino_model/resnet50 --input_shape \[1\,3\,224\,224\]
```

转换后的openvino_model/resnet50目录下会出现三个文件， 目录结构如下：

```
resnet50
├── ResNet50.bin
├── ResNet50.mapping
└── ResNet50.xml
```

**注意：**

- 留意模型转换的输出，比如转换onnx时根据提示调整opset_version的值
- paddle inference模型中的配置文件(如 `resnet50_imagenet.yml`)包含了前后处理、标签等信息，对转换后的openvino模型进行推理时还会用到。


## 编译步骤
### Step 1. 获取部署代码
```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX/deploy/cpp
```
**说明**：`C++`预测代码在`PaddleX/deploy/cpp` 目录，该目录不依赖任何`PaddleX`下其他目录。所有的推理实现代码在`model_deploy`目录下，所有示例代码都在`demo`目录下，script目录为编译脚本。

### Step 2. 修改编译参数
根据自己的系统环境，修改`PaddleX/deploy/cpp/script/openvino_build.sh`脚本中的参数，主要修改的参数为以下几个
| 参数          | 说明                                                                                 |
| :------------ | :----------------------------------------------------------------------------------- |
| OPENVINO_DIR      | OpenVINO预编译库inference_engine的路径                             |
| NGRAPH_LIB    | OpenVINO的ngraph lib的路径，编译openvino时通常会生成                   |
| GFLAGS_DIR      | gflags所在的目录路径,如果采用自动下载不用改                                            |
| OPENCV_DIR     | opencv所在的目录路径，如果采用自动下载不用改                                                       |

### Step 4. 编译
修改完openvino_build.sh后执行编译， **[注意]**: 以下命令在`PaddleX/deploy/cpp`目录下进行执行

```
sh scripts/onnx/openvino_build.sh
```
#### 编译环境无法联网导致编译失败？

> 编译过程，会联网下载opencv、gflag，如无法联网，用户按照下操作手动下载
>
> 1. 根据系统版本，点击右侧链接下载不同版本的opencv依赖 [Ubuntu 16.04](https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2)/[Ubuntu 18.04](https://bj.bcebos.com/paddlex/deploy/opencv3.4.6gcc4.8ffmpeg_ubuntu_18.04.tar.gz2)
> 2. 解压下载的opencv依赖（解压后目录名为opencv3.4.6gcc4.8ffmpeg)，创建目录`PaddleX/deploy/cpp/deps`，将解压后的目录拷贝至该创建的目录下
> 3. 点击[下载gflags依赖包](https://bj.bcebos.com/paddlex/deploy/gflags.tar.gz)，解压至`deps`目录

### Step 5. 编译结果

编译后会在`PaddleX/deploy/cpp/build/demo`目录下生成`model_infer` 可执行二进制文件示例，用于加载模型进行预测。以上面转换的ResNet50模型为例，运行指令如下：

```
./build/demo/model_infer --xml_file openvino_model/resnet50/ResNet50_vd.xml --bin_file openvino_model/resnet50/ResNet50_vd.bin --cfg_file openvino_model/resnet50/resnet50_imagenet.yml --model_type clas --image test.jpeg
```

**参数说明**

| 参数名称   | 含义                                                         |
| ---------- | ------------------------------------------------------------ |
| xml_file | openvino转换的xml模型文件                                                 |
| bin_file | openvino转换的xml模型文件                                                 |
| cfg_file   | Paddle套件导出的模型配置文件，如`resnet50/deploy.yml`    |
| image      | 需要预测的单张图片的文件路径                                 |
| model_type | 模型来源，det/seg/clas/paddlex，分别表示模型来源于PaddleDetection、PaddleSeg、PaddleClas和PaddleX |
