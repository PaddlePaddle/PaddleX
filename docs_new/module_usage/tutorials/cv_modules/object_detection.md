# 目标检测模块开发教程

## 一、概述
目标检测模块是计算机视觉系统中的关键组成部分，负责在图像或视频中定位和标记出包含特定目标的区域。该模块的性能直接影响到整个计算机视觉系统的准确性和效率。目标检测模块通常会输出目标区域的边界框（Bounding Boxes），这些边界框将作为输入传递给目标识别模块进行后续处理。

## 二、支持模型列表
<details>
   <summary> 👉模型列表详情</summary>

<table >
  <tr>
    <th>模型</th>
    <th>mAP(%)</th>
    <th>GPU推理耗时 (ms)</th>
    <th>CPU推理耗时</th>
    <th>模型存储大小 (M)</th>
    <th>介绍</th>
  </tr>
  <tr>
    <td>Cascade-FasterRCNN-ResNet50-FPN</td>
    <td>41.1</td>
    <td></td>
    <td></td>
    <td>245.4 M</td>
    <td rowspan="2">Cascade-FasterRCNN 是一种改进的Faster R-CNN目标检测模型，通过耦联多个检测器，利用不同IoU阈值优化检测结果，解决训练和预测阶段的mismatch问题，提高目标检测的准确性。</td>
  </tr>
  <tr>
    <td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
    <td>45.0</td>
    <td></td>
    <td></td>
    <td>246.2 M</td>
    <td></td>
  </tr>
  <tr>
    <td>CenterNet-DLA-34</td>
    <td>37.6</td>
    <td></td>
    <td></td>
    <td>75.4 M</td>
    <td rowspan="2">CenterNet是一种anchor-free目标检测模型，把待检测物体的关键点视为单一点-即其边界框的中心点，并通过关键点进行回归。</td>
  </tr>
  <tr>
    <td>CenterNet-ResNet50</td>
    <td>38.9</td>
    <td></td>
    <td></td>
    <td>319.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>DETR-R50</td>
    <td>42.3</td>
    <td></td>
    <td></td>
    <td>159.3 M</td>
    <td >DETR 是Facebook提出的一种transformer目标检测模型，该模型在不需要预定义的先验框anchor和NMS的后处理策略的情况下，就可以实现端到端的目标检测。</td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet34-FPN</td>
    <td>37.8</td>
    <td></td>
    <td></td>
    <td>137.5 M</td>
    <td rowspan="9">Faster R-CNN是典型的two-stage目标检测模型，即先生成区域建议（Region Proposal），然后在生成的Region Proposal上做分类和回归。相较于前代R-CNN和Fast R-CNN，Faster R-CNN的改进主要在于区域建议方面，使用区域建议网络（Region Proposal Network, RPN）提供区域建议，以取代传统选择性搜索。RPN是卷积神经网络，并与检测网络共享图像的卷积特征，减少了区域建议的计算开销。</td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-FPN</td>
    <td>38.4</td>
    <td></td>
    <td></td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-vd-FPN</td>
    <td>39.5</td>
    <td></td>
    <td></td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
    <td>41.4</td>
    <td></td>
    <td></td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50</td>
    <td>36.7</td>
    <td></td>
    <td></td>
    <td>120.2 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet101-FPN</td>
    <td>41.4</td>
    <td></td>
    <td></td>
    <td>216.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet101</td>
    <td>39.0</td>
    <td></td>
    <td></td>
    <td>188.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNeXt101-vd-FPN</td>
    <td>43.4</td>
    <td></td>
    <td></td>
    <td>360.6 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-Swin-Tiny-FPN</td>
    <td>42.6</td>
    <td></td>
    <td></td>
    <td>159.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FCOS-ResNet50</td>
    <td>39.6</td>
    <td></td>
    <td></td>
    <td>124.2 M</td>
    <td>FCOS是一种密集预测的anchor-free目标检测模型，使用RetinaNet的骨架，直接在feature map上回归目标物体的长宽，并预测物体的类别以及centerness（feature map上像素点离物体中心的偏移程度），centerness最终会作为权重来调整物体得分。</td>
  </tr>
  <tr>
    <td>PicoDet-L</td>
    <td>42.6</td>
    <td></td>
    <td></td>
    <td>20.9 M</td>
    <td rowspan="4">PP-PicoDet是一种全尺寸、棱视宽目标的轻量级目标检测算法，它考虑移动端设备运算量。与传统目标检测算法相比，PP-PicoDet具有更小的模型尺寸和更低的计算复杂度，并在保证检测精度的同时更高的速度和更低的延迟。</td>
  </tr>
  <tr>
    <td>PicoDet-M</td>
    <td>37.5</td>
    <td></td>
    <td></td>
    <td>16.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PicoDet-S</td>
    <td>29.1</td>
    <td></td>
    <td></td>
    <td>4.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PicoDet-XS</td>
    <td>26.2</td>
    <td></td>
    <td></td>
    <td>5.7 M</td>
    <td></td>
  </tr>
    <tr>
    <td>PP-YOLOE_plus-L</td>
    <td>52.9</td>
    <td></td>
    <td></td>
    <td>185.3 M</td>
    <td rowspan="4">PP-YOLOE_plus 是一种是百度飞桨视觉团队自研的动边一体高精度模型PP-YOLOE迭代优化升级的版本，通过使用Objects365大规模数据集，优化预处理、大模型扩展增强训练策略。</td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-M</td>
    <td>49.8</td>
    <td></td>
    <td></td>
    <td>82.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-S</td>
    <td>43.7</td>
    <td></td>
    <td></td>
    <td>28.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-X</td>
    <td>54.7</td>
    <td></td>
    <td></td>
    <td>349.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-H</td>
    <td>56.3</td>
    <td></td>
    <td></td>
    <td>435.8 M</td>
    <td rowspan="5">RT-DETR是第一个实时端到端目标检测器。该模型设计了一个高效的混合编码器，满足模型效果与吞吐率的双需求，高效处理多尺度特征，并提出了加速和优化的查询选择机制，以优化解码器查询的动态化。RT-DETR支持通过使用不同的解码器来实现灵活端到端推理速度。</td>
  </tr>
  <tr>
    <td>RT-DETR-L</td>
    <td>53.0</td>
    <td></td>
    <td></td>
    <td>113.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-R18</td>
    <td>46.5</td>
    <td></td>
    <td></td>
    <td>70.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-R50</td>
    <td>53.1</td>
    <td></td>
    <td></td>
    <td>149.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-X</td>
    <td>54.8</td>
    <td></td>
    <td></td>
    <td>232.9 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOv3-DarkNet53</td>
    <td>39.1</td>
    <td></td>
    <td></td>
    <td>219.7 M</td>
    <td rowspan="3">YOLOv3是一种实时的端到端目标检测器。它使用一个独特的单个卷积神经网络，将目标检测问题分解为一个回归问题，从而实现实时的检测。该模型采用了多个尺度的检测，提高了不同尺度目标物体的检测性能。</td>
  </tr>
  <tr>
    <td>YOLOv3-MobileNetV3</td>
    <td>31.4</td>
    <td></td>
    <td></td>
    <td>83.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOv3-ResNet50_vd_DCN</td>
    <td>40.6</td>
    <td></td>
    <td></td>
    <td>163.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-L</td>
    <td>50.1</td>
    <td></td>
    <td></td>
    <td>192.5 M</td>
    <td rowspan="6">YOLOX模型以YOLOv3作为目标检测网络的框架，通过设计Decoupled Head、Data Aug、Anchor Free以及SimOTA组件，显著提升了模型在各种复杂场景下的检测性能。</td>
  </tr>
  <tr>
    <td>YOLOX-M</td>
    <td>46.9</td>
    <td></td>
    <td></td>
    <td>90.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-N</td>
    <td>26.1</td>
    <td></td>
    <td></td>
    <td>3.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-S</td>
    <td>40.4</td>
    <td></td>
    <td></td>
    <td>32.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-T</td>
    <td>32.9</td>
    <td></td>
    <td></td>
    <td>18.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-X</td>
    <td>51.8</td>
    <td></td>
    <td></td>
    <td>351.5 M</td>
    <td></td>
  </tr>
</table>


**注：以上精度指标为[COCO2017](https://cocodataset.org/#home)验证集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**
</details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成目标检测模块的推理，可以任意切换该模块下的模型，您也可以将目标检测的模块中的模型推理集成到您的项目中。

```python
from paddlex import create_model
model = create_model("PicoDet-S")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的目标检测模型。在使用 PaddleX 开发目标检测模型之前，请务必安装 PaddleX的目标检测相关模型训练插件，安装过程可以参考[PaddleX本地安装教程](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/dF1VvOPZmZXXzn?t=mention&mt=doc&dt=doc)中的二次开发部分。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，**只有通过数据校验的数据才可以进行模型训练**。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX目标检测任务模块数据标注教程](../../../data_annotations/cv_modules/object_detection.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details>
  <summary>👉 <b>校验结果详情（点击展开）</b></summary>



校验结果文件具体内容为：

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 4,
    "train_samples": 701,
    "train_sample_paths": [
      "check_dataset/demo_img/road839.png",
      "check_dataset/demo_img/road363.png",
      "check_dataset/demo_img/road148.png",
      "check_dataset/demo_img/road237.png",
      "check_dataset/demo_img/road733.png",
      "check_dataset/demo_img/road861.png",
      "check_dataset/demo_img/road762.png",
      "check_dataset/demo_img/road515.png",
      "check_dataset/demo_img/road754.png",
      "check_dataset/demo_img/road173.png"
    ],
    "val_samples": 176,
    "val_sample_paths": [
      "check_dataset/demo_img/road218.png",
      "check_dataset/demo_img/road681.png",
      "check_dataset/demo_img/road138.png",
      "check_dataset/demo_img/road544.png",
      "check_dataset/demo_img/road596.png",
      "check_dataset/demo_img/road857.png",
      "check_dataset/demo_img/road203.png",
      "check_dataset/demo_img/road589.png",
      "check_dataset/demo_img/road655.png",
      "check_dataset/demo_img/road245.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/det_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```
上述校验结果中，check_pass 为 true 表示数据集格式符合要求，其他部分指标的说明如下：

* `attributes.num_classes`：该数据集类别数为 4；
* `attributes.train_samples`：该数据集训练集样本数量为 704；
* `attributes.val_samples`：该数据集验证集样本数量为 176；
* `attributes.train_sample_paths`：该数据集训练集样本可视化图片相对路径列表；
* `attributes.val_sample_paths`：该数据集验证集样本可视化图片相对路径列表；
另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）： 

![](/tmp/images/modules/obj_det/01.png)
</details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过**修改配置文件**或是**追加超参数**的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。。

<details>
  <summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

**（1）数据集格式转换**

目标检测支持 `VOC`、`LabelMe` 格式的数据集转换为 `COCO` 格式。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `convert`:
    * `enable`: 是否进行数据集格式转换，目标检测支持 `VOC`、`LabelMe` 格式的数据集转换为 `COCO` 格式，默认为 `False`;
    * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，默认为 `null`，可选值为 `VOC`、`LabelMe` 和 `VOCWithUnlabeled`、`LabelMeWithUnlabeled` ；
例如，您想转换 `LabelMe` 格式的数据集为 `COCO` 格式，以下面的`LabelMe` 格式的数据集为例，则需要修改配置如下：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_labelme_examples.tar -P ./dataset
tar -xf ./dataset/det_labelme_examples.tar -C ./dataset/
```
```bash
......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
```
随后执行命令：

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples
```
当然，以上参数同样支持通过追加命令行参数的方式进行设置，以 `LabelMe` 格式的数据集为例：

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
```
**（2）数据集划分**

数据集划分的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `split`:
    * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
    * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
    * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；
例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：

```bash
......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
```
随后执行命令：

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples 
```
数据划分执行之后，原有标注文件会被在原路径下重命名为 `xxx.bak`。

以上参数同样支持通过追加命令行参数的方式进行设置：

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处目标检测模型 `PicoDet-S` 的训练为例：

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet-S.yaml`）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>


* 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段进行设置。
* PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。
* 训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/0PKFjfhs0UN4Qs?t=mention&mt=doc&dt=doc)。
在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* `train_result.json`：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* `train.log`：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* `config.yaml`：训练配置文件，记录了本次训练的超参数的配置；
* `.pdparams`、`.pdema`、`.pdopt.pdstate`、`.pdiparams`、`.pdmodel`：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；
</details>

## **4.3 模型评估**
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet-S.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>


在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model/best_model.pdparams`。

在完成模型评估后，会产出`evaluate_result.json，其记录了`评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 AP；

</details>

### **4.4 模型推理和模型集成**
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令：

在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测。在PaddleX中实现模型推理预测可以通过两种方式：命令行和wheel 包。

* 通过命令行的方式进行推理预测，只需如下一条命令：
```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet-S.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.**产线集成**

目标检测模块可以集成的PaddleX产线有[通用目标检测产线](../../../pipeline_usage/tutorials/cv_pipelines/object_detection.md)，只需要替换模型路径即可完成相关产线的目标检测模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.**模块集成**

您产出的权重可以直接集成到目标检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。