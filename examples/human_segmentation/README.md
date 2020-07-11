# HumanSeg人像分割模型

本教程基于PaddleX核心分割模型实现人像分割，开放预训练模型和测试数据、支持视频流人像分割、提供模型Fine-tune到Paddle-Lite移动端部署的全流程应用指南。

## 目录

* [预训练模型和测试数据](#1)
* [快速体验视频流人像分割](#2)
* [模型Fine-tune](#3)
* [Paddle-Lite移动端部署](#4)


## <h2 id="1">预训练模型和测试数据</h2>

#### 预训练模型

本案例开放了两个在大规模人像数据集上训练好的模型，以满足服务器端场景和移动端场景的需求。使用这些模型可以快速体验视频流人像分割，也可以部署到移动端进行实时人像分割，也可以用于完成模型Fine-tuning。

| 模型类型 | Checkpoint Parameter | Inference Model | Quant Inference Model | 备注 |
| --- | --- | --- | ---| --- |
| HumanSeg-server  | [humanseg_server_params](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_server.pdparams) | [humanseg_server_inference](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_server_inference.zip) | -- | 高精度模型，适用于服务端GPU且背景复杂的人像场景， 模型结构为Deeplabv3+/Xcetion65, 输入大小（512， 512） |
| HumanSeg-mobile | [humanseg_mobile_params](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_mobile.pdparams) | [humanseg_mobile_inference](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_mobile_inference.zip) | [humanseg_mobile_quant](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_mobile_quant.zip) | 轻量级模型, 适用于移动端或服务端CPU的前置摄像头场景，模型结构为HRNet_w18_small_v1，输入大小（192， 192）  |

> * Checkpoint Parameter为模型权重，用于Fine-tuning场景。
> * Inference Model和Quant Inference Model为预测部署模型，包含`__model__`计算图结构、`__params__`模型参数和`model.yaml`基础的模型配置信息。
> * 其中Inference Model适用于服务端的CPU和GPU预测部署，Qunat Inference Model为量化版本，适用于通过Paddle Lite进行移动端等端侧设备部署。


预训练模型的存储大小和推理时长如下所示，其中移动端模型的运行环境为cpu：骁龙855，内存：6GB，图片大小：192*192

| 模型 | 模型大小 | 计算耗时 |
| --- | --- | --- |
|humanseg_server_inference| 158M | - |
|humanseg_mobile_inference | 5.8 M | 42.35ms |
|humanseg_mobile_quant | 1.6M | 24.93ms |

执行以下脚本下载全部的预训练模型：
```bash
python pretrain_weights/download_pretrain_weights.py
```

#### 测试数据

[supervise.ly](https://supervise.ly/)发布了人像分割数据集**Supervisely Persons**, 本案例从中随机抽取一小部分数据并转化成PaddleX可直接加载的数据格式，运行以下代码可下载该数据、以及手机前置摄像头拍摄的人像测试视频`video_test.mp4`.

```bash
python data/download_data.py
```

## <h2 id="2">快速体验视频流人像分割</h2>

#### 前置依赖

* PaddlePaddle >= 1.8.0
* Python >= 3.5
* PaddleX >= 1.0.0

安装的相关问题参考[PaddleX安装](../../docs/install.md)

### 光流跟踪辅助的视频流人像分割

本案例将DIS（Dense Inverse Search-basedmethod）光流跟踪算法的预测结果与PaddleX的分割结果进行融合，以此改善视频流人像分割的效果。运行以下代码进行体验：

* 通过电脑摄像头进行实时分割处理

```bash
python video_infer.py --model_dir pretrain_weights/humanseg_mobile_inference
```
* 对离线人像视频进行分割处理

```bash
python video_infer.py --model_dir pretrain_weights/humanseg_mobile_inference --video_path data/video_test.mp4
```

视频分割结果如下所示：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/result.gif" width="20%" height="20%">

### 人像背景替换

本案例还实现了人像背景替换功能，根据所选背景对人像的背景画面进行替换，背景可以是一张图片，也可以是一段视频。

* 通过电脑摄像头进行实时背景替换处理, 通过'--background_video_path'传入背景视频
```bash
python bg_replace.py --model_dir pretrain_weights/humanseg_mobile_inference --background_image_path data/background.jpg
```

* 对人像视频进行背景替换处理, 通过'--background_video_path'传入背景视频
```bash
python bg_replace.py --model_dir pretrain_weights/humanseg_mobile_inference --video_path data/video_test.mp4 --background_image_path data/background.jpg
```

* 对单张图像进行背景替换
```bash
python bg_replace.py --model_dir pretrain_weights/humanseg_mobile_inference --image_path data/human_image.jpg --background_image_path data/background.jpg
```

背景替换结果如下：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/bg_replace.gif" width="20%" height="20%">

**注意**:

* 视频分割处理时间需要几分钟，请耐心等待。

* 提供的模型适用于手机摄像头竖屏拍摄场景，宽屏效果会略差一些。

## <h2 id="3">模型Fine-tune</h2>

#### 前置依赖

* PaddlePaddle >= 1.8.0
* Python >= 3.5
* PaddleX >= 1.0.0

安装的相关问题参考[PaddleX安装](../../docs/install.md)

### 模型训练

使用下述命令进行基于预训练模型的模型训练，请确保选用的模型结构`model_type`与模型参数`pretrain_weights`匹配。如果不需要本案例提供的测试数据，可更换数据、选择合适的模型并调整训练参数。

```bash
# 指定GPU卡号（以0号卡为例）
export CUDA_VISIBLE_DEVICES=0
# 若不使用GPU，则将CUDA_VISIBLE_DEVICES指定为空
# export CUDA_VISIBLE_DEVICES=
python train.py --model_type HumanSegMobile \
--save_dir output/ \
--data_dir data/mini_supervisely \
--train_list data/mini_supervisely/train.txt \
--val_list data/mini_supervisely/val.txt \
--pretrain_weights pretrain_weights/humanseg_mobile_params \
--batch_size 8 \
--learning_rate 0.001 \
--num_epochs 10 \
--image_shape 192 192
```
其中参数含义如下：
* `--model_type`: 模型类型，可选项为：HumanSegServer和HumanSegMobile
* `--save_dir`: 模型保存路径
* `--data_dir`: 数据集路径
* `--train_list`: 训练集列表路径
* `--val_list`: 验证集列表路径
* `--pretrain_weights`: 预训练模型路径
* `--batch_size`: 批大小
* `--learning_rate`: 初始学习率
* `--num_epochs`: 训练轮数
* `--image_shape`: 网络输入图像大小（w, h）

更多命令行帮助可运行下述命令进行查看：
```bash
python train.py --help
```
**注意**：可以通过更换`--model_type`变量与对应的`--pretrain_weights`使用不同的模型快速尝试。

### 评估

使用下述命令对模型在验证集上的精度进行评估：

```bash
python eval.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--val_list data/mini_supervisely/val.txt \
--image_shape 192 192
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--val_list`: 验证集列表路径
* `--image_shape`: 网络输入图像大小（w, h）

### 预测

使用下述命令对测试集进行预测，预测可视化结果默认保存在`./output/result/`文件夹中。
```bash
python infer.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--test_list data/mini_supervisely/test.txt \
--save_dir output/result \
--image_shape 192 192
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--test_list`: 测试集列表路径
* `--image_shape`: 网络输入图像大小（w, h）

### 模型导出

在服务端部署的模型需要首先将模型导出为inference格式模型，导出的模型将包括`__model__`、`__params__`和`model.yml`三个文名，分别为模型的网络结构，模型权重和模型的配置文件（包括数据预处理参数等等）。在安装完PaddleX后，在命令行终端使用如下命令完成模型导出：

```bash
paddlex --export_inference --model_dir output/best_model \
--save_dir output/export
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--save_dir`: 导出模型保存路径

### 离线量化
```bash
python quant_offline.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--quant_list data/mini_supervisely/val.txt \
--save_dir output/quant_offline \
--image_shape 192 192
```
其中参数含义如下：
* `--model_dir`: 待量化模型路径
* `--data_dir`: 数据集路径
* `--quant_list`: 量化数据集列表路径，一般直接选择训练集或验证集
* `--save_dir`: 量化模型保存路径
* `--image_shape`: 网络输入图像大小（w, h）

## <h2 id="4">Paddle-Lite移动端部署</h2>
