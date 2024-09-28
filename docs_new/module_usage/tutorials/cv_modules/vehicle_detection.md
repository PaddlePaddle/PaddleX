# 车辆检测模块开发教程

## 一、概述
车辆检测是目标检测的一个子任务，具体指使用计算机视觉技术来判断图像或视频中是否存在车辆，并给出每个车辆的具体位置信息（如边界框的坐标）。这些信息对于智能交通系统、自动驾驶、视频监控等多个领域具有重要意义。

## 二、支持模型列表

<details>
   <summary> 👉模型列表详情</summary>

<table>
  <tr>
    <th>模型</th>
    <th>mAP 0.5:0.95</th>
    <th>GPU推理耗时（ms）</th>
    <th>CPU推理耗时</th>
    <th>模型存储大小（M）</th>
    <th>介绍</th>
  </tr>
  <tr>
    <td>PP-YOLOE-S_vehicle</td>
    <td>61.3</td>
    <td>15.4</td>
    <td>178.4</td>
    <td>28.79</td>
    <td rowspan="2">基于PP-YOLOE的行人检测模型</td>
  </tr>
  <tr>
    <td>PP-YOLOE-L_vehicle</td>
    <td>63.9</td>
    <td>32.6</td>
    <td>775.6</td>
    <td>196.02</td>
  </tr>
</table>

注：以上精度指标为PPVehicle 验证集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。
</details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成wheel包的安装后，几行代码即可完成车辆检测模块的推理，可以任意切换该模块下的模型，您也可以将车辆检测的模块中的模型推理集成到您的项目中。

```python
from paddlex.inference import create_model 

model_name = "PP-YOLOE-S_vehicle"

model = create_model(model_name)
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_detection.jpg", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")

```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用PaddleX的二次开发能力，开发更好的车辆检测模型。在使用PaddleX开发车辆检测模型之前，请务必安装PaddleX的PaddleDetection插件，安装过程可以参考[PaddleX本地安装教程](../../../installation/installation.md)。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，**只有通过数据校验的数据才可以进行模型训练**。此外，PaddleX为每一个模块都提供了demo数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX目标检测任务模块数据标注教程](../../../data_annotations/cv_modules/object_detection.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/vehicle_coco_examples.tar -P ./dataset
tar -xf ./dataset/vehicle_coco_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
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
    "train_samples": 500,
    "train_sample_paths": [
      "check_dataset/demo_img/MVI_20011__img00001.jpg",
      "check_dataset/demo_img/MVI_20011__img00005.jpg",
      "check_dataset/demo_img/MVI_20011__img00009.jpg"
    ],
    "val_samples": 100,
    "val_sample_paths": [
      "check_dataset/demo_img/MVI_20032__img00401.jpg",
      "check_dataset/demo_img/MVI_20032__img00405.jpg",
      "check_dataset/demo_img/MVI_20032__img00409.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/example_data/vehicle_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```
上述校验结果中，`check_pass` 为 `True` 表示数据集格式符合要求，其他部分指标的说明如下：

* `attributes.num_classes`：该数据集类别数为4；
* `attributes.train_samples`：该数据集训练集样本数量为 500；
* `attributes.val_samples`：该数据集验证集样本数量为 100；
* `attributes.train_sample_paths`：该数据集训练集样本可视化图片相对路径列表；
* `attributes.val_sample_paths`：该数据集验证集样本可视化图片相对路径列表；


数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）： 

![](/tmp/images/modules/vehicle_det/01.png)
</details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过**修改配置文件**或是**追加超参数**的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details>
  <summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>


**（1）数据集格式转换**

人脸检测不支持数据格式转换。

**（2）数据集划分**

数据集划分的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `split`:
    * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
    * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证与 `val_percent` 的值之和为100；


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
python main.py -c paddlex/configs/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
```
数据划分执行之后，原有标注文件会被在原路径下重命名为 `xxx.bak`。

以上参数同样支持通过追加命令行参数的方式进行设置：

```bash
python main.py -c paddlex/configs/vehicle_detection/PP-YOLOE-S_vehicle.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处`PP-YOLOE-S_vehicle`的训练为例：

```bash
python main.py -c paddlex/configs/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-S_vehicle.yaml`）
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
python main.py -c paddlex/configs/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-S_vehicle``.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>


在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=``./output/best_model/best_model/model.pdparams`。

在完成模型评估后，会产出`evaluate_result.json，其记录了`评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 AP；

</details>

### **4.4 模型推理**
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测。在PaddleX中实现模型推理预测可以通过两种方式：命令行和wheel 包。

#### 4.4.1 模型推理
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.**产线集成**

目标检测模块可以集成的PaddleX产线有[通用目标检测产线](../../../pipeline_usage/tutorials/cv_pipelines/object_detection.md)，只需要替换模型路径即可完成相关产线的目标检测模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.**模块集成**

您产出的权重可以直接集成到目标检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。

* 通过命令行的方式进行推理预测，只需如下一条命令：
```bash
python main.py -c paddlex/configs/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_detection.jpg"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-S_vehicle.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
您产出的权重可以直接集成到车辆检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
