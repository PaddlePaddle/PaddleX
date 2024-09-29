# 图像多标签分类模块开发教程

## 一、概述
图像多标签分类模块是计算机视觉系统中的重要组件，负责对输入的图像进行多标签的分类。与传统的图像分类任务只将图像分到一个类别不同，图像多标签分类任务需要将图像分到多个相关的类别。该模块的性能直接影响到整个计算机视觉系统的准确性和效率。图像多标签分类模块通常会接收图像作为输入，然后通过深度学习或其他机器学习算法，根据图像的特性和内容，将其分类到多个预定义的类别中。例如，对于一张包含猫和狗的图像，图像多标签分类模块可能需要将其同时标记为“猫”和“狗”。这些分类标签将作为输出，供其他模块或系统进行后续的处理和分析。

## 二、支持模型列表

<details>
   <summary> 👉模型列表详情</summary>

<table>
  <tr>
    <th>模型</th>
    <th>mAP(%)</th>
    <th>模型存储大小 (M)</th>
    <th>介绍</th>
  </tr>
  <tr>
    <td>CLIP_vit_base_patch16_448_ML</td>
    <td>89.15</td>
    <td>325.6</td>
    <td>CLIP_ML是一种基于CLIP的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B0_ML</td>
    <td>80.98</td>
    <td>39.6</td>
    <td rowspan="3">PP-HGNetV2_ML是一种基于PP-HGNetV2的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B4_ML</td>
    <td>87.96</td>
    <td>88.5</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B6_ML</td>
    <td>91.25</td>
    <td>286.0</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_0_ML</td>
    <td>77.96</td>
    <td>29.4</td>
    <td>PP-LCNet_ML是一种基于PP-LCNet的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
  </tr>
  <tr>
    <td>ResNet50_ML</td>
    <td>83.50</td>
    <td>108.9</td>
    <td>ResNet50_ML是一种基于ResNet50的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
  </tr>
</table>


**注：以上精度指标为[COCO2017](https://cocodataset.org/#home)的多标签分类任务mAP。**
</details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

wheel 包的安装后，几行代码即可完成图像多标签分类模块的推理，可以任意切换该模块下的模型，您也可以将图像多标签分类的模块中的模型推理集成到您的项目中。

```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0_ML")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的图像多标签分类模型。在使用 PaddleX 开发图像多标签分类模型之前，请务必安装 PaddleX 的 图像分类 相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，**只有通过数据校验的数据才可以进行模型训练**。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX多标签分类任务模块数据标注教程](../../../data_annotations/cv_modules/ml_classification.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/mlcls_nus_examples.tar -P ./dataset
tar -xf ./dataset/mlcls_nus_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在 log 中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details>
  <summary>👉 <b>校验结果详情（点击展开）</b></summary>

校验结果文件具体内容为：

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "../../dataset/mlcls_nus_examples/label.txt",
    "num_classes": 33,
    "train_samples": 17463,
    "train_sample_paths": [
      "check_dataset/demo_img/0543_4338693.jpg",
      "check_dataset/demo_img/0272_347806939.jpg",
      "check_dataset/demo_img/0069_2291994812.jpg",
      "check_dataset/demo_img/0012_1222850604.jpg",
      "check_dataset/demo_img/0238_53773041.jpg",
      "check_dataset/demo_img/0373_541261977.jpg",
      "check_dataset/demo_img/0567_519506868.jpg",
      "check_dataset/demo_img/0023_289621557.jpg",
      "check_dataset/demo_img/0581_484524659.jpg",
      "check_dataset/demo_img/0325_120753036.jpg"
    ],
    "val_samples": 17463,
    "val_sample_paths": [
      "check_dataset/demo_img/0546_130758157.jpg",
      "check_dataset/demo_img/0284_2230710138.jpg",
      "check_dataset/demo_img/0090_1491261559.jpg",
      "check_dataset/demo_img/0013_392798436.jpg",
      "check_dataset/demo_img/0246_2248376356.jpg",
      "check_dataset/demo_img/0377_1349296474.jpg",
      "check_dataset/demo_img/0570_2457645006.jpg",
      "check_dataset/demo_img/0027_309333946.jpg",
      "check_dataset/demo_img/0584_132639537.jpg",
      "check_dataset/demo_img/0329_206031527.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/mlcls_nus_examples",
  "show_type": "image",
  "dataset_type": "MLClsDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

* `attributes.num_classes`：该数据集类别数为 33；
* `attributes.train_samples`：该数据集训练集样本数量为 17463；
* `attributes.val_samples`：该数据集验证集样本数量为 17463；
* `attributes.train_sample_paths`：该数据集训练集样本可视化图片相对路径列表；
* `attributes.val_sample_paths`：该数据集验证集样本可视化图片相对路径列表；


另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）： 

![](/tmp/images/modules/ml_classification/01.png)
</details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过**修改配置文件**或是**追加超参数**的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。。

<details>
  <summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

**（1）数据集格式转换**

图像多标签分类支持 `COCO`格式的数据集转换为 `MLClsDataset`格式，数据集格式转换的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `convert`:
    * `enable`: 是否进行数据集格式转换，图像多标签分类支持 `COCO`格式的数据集转换为 `MLClsDataset`格式，默认为 `False`;
    * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，默认为 `null`，可选值为 `COCO` ；
  
例如，您想将`COCO`格式的数据集转换为 `MLClsDataset`格式，则需将配置文件修改为：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```
```bash
......
CheckDataset:
  ......
  convert: 
    enable: True
    src_dataset_type: COCO
  ......
```
随后执行命令：

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples 
```
数据转换执行之后，原有标注文件会被在原路径下重命名为 `xxx.bak`。

以上参数同样支持通过追加命令行参数的方式进行设置：

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=COCO
```
**（2）数据集划分**

数据集划分的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `split`:
    * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
    * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 `val_percent` 值加和为 100；
    * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为 0-100 之间的任意整数，需要保证和 `train_percent` 值加和为 100；

例如，您想重新划分数据集为 训练集占比 90%、验证集占比 10%，则需将配置文件修改为：

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
一条命令即可完成模型的训练，以此处图像多标签分类模型 PP-LCNet_x1_0_ML 的训练为例：

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0_ML.yaml`）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>

* 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段进行设置。
* PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。
* 训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)。
在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* `train_result.json`：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* `train.log`：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* `config.yaml`：训练配置文件，记录了本次训练的超参数的配置；
* `.pdparams`、`.pdema`、`.pdopt.pdstate`、`.pdiparams`、`.pdmodel`：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；
</details>

## **4.3 模型评估**
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0_ML.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>

在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model/best_model.pdparams`。

在完成模型评估后，会产出`evaluate_result.json，其记录了`评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包括 MultiLabelMAP；

</details>

### **4.4 模型推理和模型集成**
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理

* 通过命令行的方式进行推理预测，只需如下一条命令：
```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0_ML.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.**产线集成**

图像多标签分类模块可以集成的PaddleX产线有[通用图像多标签分类产线](../../../pipeline_usage/tutorials/cv_pipelines/image_multi_label_lassification.md)，只需要替换模型路径即可完成相关产线的图像多标签分类模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.**模块集成**

您产出的权重可以直接集成到图像多标签分类模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
