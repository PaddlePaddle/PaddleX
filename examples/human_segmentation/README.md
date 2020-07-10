# HumanSeg人像分割模型

本教程基于PaddleX核心分割网络，提供针对人像分割场景从预训练模型、Fine-tune、视频分割预测部署的全流程应用指南。

## 安装

**前置依赖**
* paddlepaddle >= 1.8.0
* python >= 3.5

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```
安装的相关问题参考[PaddleX安装](https://paddlex.readthedocs.io/zh_CN/latest/install.html)

## 预训练模型
HumanSeg开放了在大规模人像数据上训练的两个预训练模型，满足多种使用场景的需求

| 模型类型 | Checkpoint Parameter | Inference Model | Quant Inference Model | 备注 |
| --- | --- | --- | ---| --- |
| HumanSeg-server  | [humanseg_server_params](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_server.pdparams) | [humanseg_server_inference](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_server_inference.zip) | -- | 高精度模型，适用于服务端GPU且背景复杂的人像场景， 模型结构为Deeplabv3+/Xcetion65, 输入大小（512， 512） |
| HumanSeg-mobile | [humanseg_mobile_params](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_mobile.pdparams) | [humanseg_mobile_inference](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_mobile_inference.zip) | [humanseg_mobile_quant](https://paddlex.bj.bcebos.com/humanseg/models/humanseg_mobile_quant.zip) | 轻量级模型, 适用于移动端或服务端CPU的前置摄像头场景，模型结构为HRNet_w18_samll_v1，输入大小（192， 192）  |


模型性能

| 模型 | 模型大小 | 计算耗时 |
| --- | --- | --- |
|humanseg_server_inference| 158M | - |
|humanseg_mobile_inference | 5.8 M | 42.35ms |
|humanseg_mobile_quant | 1.6M | 24.93ms |

计算耗时运行环境： 小米，cpu：骁龙855， 内存：6GB， 图片大小：192*192


**NOTE:**
其中Checkpoint Parameter为模型权重，用于Fine-tuning场景。

* Inference Model和Quant Inference Model为预测部署模型，包含`__model__`计算图结构、`__params__`模型参数和`model.yaml`基础的模型配置信息。

* 其中Inference Model适用于服务端的CPU和GPU预测部署，Qunat Inference Model为量化版本，适用于通过Paddle Lite进行移动端等端侧设备部署。

执行以下脚本进行HumanSeg预训练模型的下载
```bash
python pretrain_weights/download_pretrain_weights.py
```

## 下载测试数据
我们提供了[supervise.ly](https://supervise.ly/)发布人像分割数据集**Supervisely Persons**, 从中随机抽取一小部分并转化成PaddleX可直接加载数据格式。通过运行以下代码进行快速下载，其中包含手机前置摄像头的人像测试视频`video_test.mp4`.

```bash
python data/download_data.py
```

## 快速体验视频流人像分割
结合DIS（Dense Inverse Search-basedmethod）光流算法预测结果与分割结果，改善视频流人像分割
```bash
# 通过电脑摄像头进行实时分割处理
python video_infer.py --model_dir pretrain_weights/humanseg_mobile_inference

# 对人像视频进行分割处理
python video_infer.py --model_dir pretrain_weights/humanseg_mobile_inference --video_path data/video_test.mp4
```

视频分割结果如下：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/result.gif" width="20%" height="20%">

根据所选背景进行背景替换，背景可以是一张图片，也可以是一段视频。
```bash
# 通过电脑摄像头进行实时背景替换处理, 也可通过'--background_video_path'传入背景视频
python bg_replace.py --model_dir pretrain_weights/humanseg_mobile_inference --background_image_path data/background.jpg

# 对人像视频进行背景替换处理, 也可通过'--background_video_path'传入背景视频
python bg_replace.py --model_dir pretrain_weights/humanseg_mobile_inference --video_path data/video_test.mp4 --background_image_path data/background.jpg

# 对单张图像进行背景替换
python bg_replace.py --model_dir pretrain_weights/humanseg_mobile_inference --image_path data/human_image.jpg --background_image_path data/background.jpg

```

背景替换结果如下：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/bg_replace.gif" width="20%" height="20%">


**NOTE**:

视频分割处理时间需要几分钟，请耐心等待。

提供的模型适用于手机摄像头竖屏拍摄场景，宽屏效果会略差一些。

## 训练
使用下述命令基于与训练模型进行Fine-tuning，请确保选用的模型结构`model_type`与模型参数`pretrain_weights`匹配。
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
**NOTE**
可通过更换`--model_type`变量与对应的`--pretrain_weights`使用不同的模型快速尝试。

## 评估
使用下述命令进行评估
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

## 预测
使用下述命令进行预测， 预测结果默认保存在`./output/result/`文件夹中。
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

## 模型导出
```bash
paddlex --export_inference --model_dir output/best_model \
--save_dir output/export
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--save_dir`: 导出模型保存路径

## 离线量化
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
