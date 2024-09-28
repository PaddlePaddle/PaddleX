# 图像分类模块开发教程

## 一、概述
图像分类模块是计算机视觉系统中的关键组成部分，负责对输入的图像进行分类。该模块的性能直接影响到整个计算机视觉系统的准确性和效率。图像分类模块通常会接收图像作为输入，然后通过深度学习或其他机器学习算法，根据图像的特性和内容，将其分类到预定义的类别中。例如，对于一个动物识别系统，图像分类模块可能需要将输入的图像分类为“猫”、“狗”、“马”等类别。图像分类模块的分类结果将作为输出，供其他模块或系统使用。

## 二、支持模型列表
<details>
   <summary> 👉模型列表详情</summary>

<table>
  <tr>
    <th>模型</th>
    <th>Top1 Acc(%)</th>
    <th>GPU推理耗时 (ms)</th>
    <th>CPU推理耗时</th>
    <th>模型存储大小 (M)</th>
    <th>介绍</th>
  </tr>
  <tr>
    <td>CLIP_vit_base_patch16_224</td>
    <td>85.36</td>
    <td></td>
    <td></td>
    <td >306.5 M</td>
    <td rowspan="2">CLIP是一种基于视觉和语言相关联的图像分类模型，采用对比学习和预训练方法，实现无监督或弱监督的图像分类，尤其适用于大规模数据集。模型通过将图像和文本映射到同一表示空间，学习到通用特征，具有良好的泛化能力和解释性。其在较好的训练误差，在很多下游任务都有较好的表现。</td>
  </tr>
  <tr>
    <td>CLIP_vit_large_patch14_224</td>
    <td>88.1</td>
    <td></td>
    <td></td>
    <td>1.04 G</td>
  </tr>
  <tr>
    <td>ConvNeXt_base_224</td>
    <td>83.84</td>
    <td></td>
    <td></td>
    <td>313.9 M</td>
    <td rowspan="6">ConvNeXt系列模型是Meta在2022年提出的基于CNN架构的模型。该系列模型是在ResNet的基础上，通过借鉴SwinTransformer的优点设计，包括训练策略和网络结构的优化思路，从而改进的纯CNN架构网络，探索了卷积神经网络的性能上限。ConvNeXt系列模型具备卷积神经网络的诸多优点，包括推理效率高和易于迁移到下游任务等。</td>
  </tr>
  <tr>
    <td>ConvNeXt_base_384</td>
    <td>84.90</td>
    <td></td>
    <td></td>
    <td>313.9 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_large_224</td>
    <td>84.26</td>
    <td></td>
    <td></td>
    <td>700.7 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_large_384</td>
    <td>85.27</td>
    <td></td>
    <td></td>
    <td>700.7 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_small</td>
    <td>83.13</td>
    <td></td>
    <td></td>
    <td>178.0 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_tiny</td>
    <td>82.03</td>
    <td></td>
    <td></td>
    <td>104.1 M</td>
  </tr>
  <tr>
    <td>FasterNet-L</td>
    <td>83.5</td>
    <td></td>
    <td></td>
    <td>357.1 M</td>
    <td rowspan="6">FasterNet是一个旨在提高运行速度的神经网络，改进点主要如下：<br>
      1.重新审视了流行的运算符，发现低FLOPS主要来自于运算频繁的内存访问，特别是深度卷积；<br>
      2.提出了部分卷积(PConv)，通过减少冗余计算和内存访问来更高效地提取图像特征；<br>
      3.基于PConv推出了FasterNet系列模型，这是一种新的设计方案，在不影响模型任务性能的情况下，在各种设备上实现了显著更高的运行速度。</td>
  </tr>
  <tr>
    <td>FasterNet-M</td>
    <td>83.0</td>
    <td></td>
    <td></td>
    <td>204.6 M</td>
  </tr>
  <tr>
    <td>FasterNet-S</td>
    <td>81.3</td>
    <td></td>
    <td></td>
    <td>119.3 M</td>
  </tr>
  <tr>
    <td>FasterNet-T0</td>
    <td>71.9</td>
    <td></td>
    <td></td>
    <td>15.1 M</td>
  </tr>
  <tr>
    <td>FasterNet-T1</td>
    <td>75.9</td>
    <td></td>
    <td></td>
    <td>29.2 M</td>
  </tr>
  <tr>
    <td>FasterNet-T2</td>
    <td>79.1</td>
    <td></td>
    <td></td>
    <td>57.4 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_5</td>
    <td>63.5</td>
    <td></td>
    <td></td>
    <td>4.8 M</td>
    <td rowspan="4">MobileNetV1是Google于2017年发布的用于移动设备或嵌入式设备中的网络。该网络将传统的卷积操作拆解成深度可分离卷积，即Depthwise卷积和Pointwise卷积的组合。相比传统的卷积网络，该组合可以大大节省参数量和计算量。同时该网络可以用于图像分类等其他视觉任务中。</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_25</td>
    <td>51.4</td>
    <td></td>
    <td></td>
    <td>1.8 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_75</td>
    <td>68.8</td>
    <td></td>
    <td></td>
    <td>9.3 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x1_0</td>
    <td>71.0</td>
    <td></td>
    <td></td>
    <td>15.2 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x0_5</td>
    <td>65.0</td>
    <td></td>
    <td></td>
    <td>7.1 M</td>
    <td rowspan="5">MobileNetV2是Google继MobileNetV1提出的一种轻量级网络。相比MobileNetV1，MobileNetV2提出了Linear bottlenecks与Inverted residual block作为网络基本结构，通过大量地堆叠这些基本模块，构成了MobileNetV2的网络结构。最后，在FLOPs只有MobileNetV1的一半的情况下取得了更高的分类精度。</td>
  </tr>
  <tr>
    <td>MobileNetV2_x0_25</td>
    <td>53.2</td>
    <td></td>
    <td></td>
    <td>5.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x1_0</td>
    <td>72.2</td>
    <td></td>
    <td></td>
    <td>12.6 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x1_5</td>
    <td>74.1</td>
    <td></td>
    <td></td>
    <td>25.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x2_0</td>
    <td>75.2</td>
    <td></td>
    <td></td>
    <td>41.2 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_5</td>
    <td>69.2</td>
    <td></td>
    <td></td>
    <td>9.6 M</td>
    <td rowspan="10">MobileNetV3是Google于2019年提出的一种基于NAS的轻量级网络。为了进一步提升效果，将relu和sigmoid激活函数分别替换为hard_swish与hard_sigmoid激活函数，同时引入了一些专门为减少网络计算量的改进策略。</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_35</td>
    <td>64.3</td>
    <td></td>
    <td></td>
    <td>7.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_75</td>
    <td>73.1</td>
    <td></td>
    <td></td>
    <td>14.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x1_0</td>
    <td>75.3</td>
    <td></td>
    <td></td>
    <td>19.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x1_25</td>
    <td>76.4</td>
    <td></td>
    <td></td>
    <td>26.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_5</td>
    <td>59.2</td>
    <td></td>
    <td></td>
    <td>6.8 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_35</td>
    <td>53.0</td>
    <td></td>
    <td></td>
    <td>6.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_75</td>
    <td>66.0</td>
    <td></td>
    <td></td>
    <td>8.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_0</td>
    <td>68.2</td>
    <td></td>
    <td></td>
    <td>10.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_25</td>
    <td>70.7</td>
    <td></td>
    <td></td>
    <td>13.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_large</td>
    <td>83.4</td>
    <td></td>
    <td></td>
    <td>125.2 M</td>
    <td rowspan="5">MobileNetV4是专为移动设备设计的高效架构。其核心在于引入了UIB（Universal Inverted Bottleneck）模块，这是一种统一且灵活的结构，融合了IB（Inverted Bottleneck）、ConvNeXt、FFN（Feed Forward Network）以及最新的ExtraDW（Extra Depthwise）模块。与UIB同时推出的还有Mobile MQA，这是种专为移动加速器定制的注意力块，可实现高达39%的显著加速。此外，MobileNetV4引入了一种新的神经架构搜索（Neural Architecture Search, NAS）方案，以提升搜索的有效性。</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_medium</td>
    <td>79.9</td>
    <td></td>
    <td></td>
    <td>37.6 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_small</td>
    <td>74.6</td>
    <td></td>
    <td></td>
    <td>14.7 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_hybrid_large</td>
    <td>83.8</td>
    <td></td>
    <td></td>
    <td>145.1 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_hybrid_medium</td>
    <td>80.5</td>
    <td></td>
    <td></td>
    <td>42.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNet_base</td>
    <td>85.0</td>
    <td></td>
    <td></td>
    <td>249.4 M</td>
    <td rowspan="3">PP-HGNet（High Performance GPU Net）是百度飞桨视觉团队研发的适用于GPU平台的高性能骨干网络。该网络结合VOVNet的基础出使用了可学习的下采样层（LDS Layer），融合了ResNet_vd、PPHGNet等模型的优点。该模型在GPU平台上与其他SOTA模型在相同的速度下有着更高的精度。在同等速度下，该模型高于ResNet34-0模型3.8个百分点，高于ResNet50-0模型2.4个百分点，在使用相同的SLSD条款下，最终超越了ResNet50-D模型4.7个百分点。与此同时，在相同精度下，其推理速度也远超主流VisionTransformer的推理速度。</td>
  </tr>
  <tr>
    <td>PP-HGNet_small</td>
    <td>81.51</td>
    <td></td>
    <td></td>
    <td>86.5 M</td>
  </tr>
  <tr>
    <td>PP-HGNet_tiny</td>
    <td>79.83</td>
    <td></td>
    <td></td>
    <td>52.4 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B0</td>
    <td>77.77</td>
    <td></td>
    <td></td>
    <td>21.4 M</td>
    <td rowspan="7">PP-HGNetV2（High Performance GPU Network V2）是百度飞桨视觉团队的PP-HGNet的下一代版本，其在PP-HGNet的基础上，做了进一步优化和改进，其在NVIDIA发布的“Accuracy-Latency Balance”做到了极致，精度大幅超越了其他同样推理速度的模型。在每种标签分类，考标场景中，都有较强的表现。</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B1</td>
    <td>79.18</td>
    <td></td>
    <td></td>
    <td>22.6 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B2</td>
    <td>81.74</td>
    <td></td>
    <td></td>
    <td>39.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B3</td>
    <td>82.98</td>
    <td></td>
    <td></td>
    <td>57.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B4</td>
    <td>83.57</td>
    <td></td>
    <td></td>
    <td>70.4 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B5</td>
    <td>84.75</td>
    <td></td>
    <td></td>
    <td>140.8 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B6</td>
    <td>86.30</td>
    <td></td>
    <td></td>
    <td>268.4 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_5</td>
    <td>63.14</td>
    <td></td>
    <td></td>
    <td>6.7 M</td>
    <td rowspan="8">PP-LCNet是百度飞桨视觉团队自研的轻量级骨干网络，它能在不增加推理时间的前提下，进一步提升模型的性能，大幅超越其他轻量级SOTA模型。</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_25</td>
    <td>51.86</td>
    <td></td>
    <td></td>
    <td>5.5 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_35</td>
    <td>58.09</td>
    <td></td>
    <td></td>
    <td>5.9 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_75</td>
    <td>68.18</td>
    <td></td>
    <td></td>
    <td>8.4 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_0</td>
    <td>71.32</td>
    <td></td>
    <td></td>
    <td>10.5 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_5</td>
    <td>73.71</td>
    <td></td>
    <td></td>
    <td>16.0 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x2_0</td>
    <td>75.18</td>
    <td></td>
    <td></td>
    <td>23.2 M</td>
  </tr>
  <tr>

  
</table>


**注：以上精度指标为 [ImageNet-1k](https://www.image-net.org/index.php) 验证集 Top1 Acc。****所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**
</details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)。

完成 wheel 包的安装后，几行代码即可完成图像分类模块的推理，可以任意切换该模块下的模型，您也可以将图像分类的模块中的模型推理集成到您的项目中。

```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的图像分类模型。在使用 PaddleX 开发图像分类模型之前，请务必安装 PaddleX 的 图像分类 相关模型训练插件，安装过程可以参考[PaddleX本地安装教程](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/dF1VvOPZmZXXzn?t=mention&mt=doc&dt=doc)中的二次开发部分。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，**只有通过数据校验的数据才可以进行模型训练**。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX图像分类任务模块数据标注教程](../../../data_annotations/cv_modules/image_classification.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details>
  <summary>👉 <b>校验结果详情（点击展开）</b></summary>
校验结果文件具体内容为：

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "dataset/label.txt",
    "num_classes": 102,
    "train_samples": 1020,
    "train_sample_paths": [
      "check_dataset/demo_img/image_01904.jpg",
      "check_dataset/demo_img/image_06940.jpg"
    ],
    "val_samples": 1020,
    "val_sample_paths": [
      "check_dataset/demo_img/image_01937.jpg",
      "check_dataset/demo_img/image_06958.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/cls_flowers_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

* `attributes.num_classes`：该数据集类别数为 102；
* `attributes.train_samples`：该数据集训练集样本数量为 1020；
* `attributes.val_samples`：该数据集验证集样本数量为 1020；
* `attributes.train_sample_paths`：该数据集训练集样本可视化图片相对路径列表；
* `attributes.val_sample_paths`：该数据集验证集样本可视化图片相对路径列表；


另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）： 

![](/tmp/images/modules/image_classification/01.png)
</details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过**修改配置文件**或是**追加超参数**的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。。

<details>
  <summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

**（1）数据集格式转换**

图像分类暂不支持数据转换。

**（2）数据集划分**

数据集划分的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `split`:
    * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
    * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 `val_percent` 值加和为100；


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
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
数据划分执行之后，原有标注文件会被在原路径下重命名为 `xxx.bak`。

以上参数同样支持通过追加命令行参数的方式进行设置：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处图像分类模型 PP-LCNet_x1_0 的训练为例：

```
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
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
python main.py -c  paddlex/configs/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>

在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model/best_model.pdparams`。

在完成模型评估后，会产出`evaluate_result.json，其记录了`评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 val.top1、val.top5；

</details>

### **4.4 模型推理和模型集成**
#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.**产线集成**

图像分类模块可以集成的 PaddleX 产线有[通用图像分类产线](../../../pipeline_usage/tutorials/cv_pipelines/image_classification.md)，只需要替换模型路径即可完成相关产线的图像分类模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.**模块集成**

您产出的权重可以直接集成到图像分类模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。