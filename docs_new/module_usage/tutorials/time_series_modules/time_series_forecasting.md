# 时序预测模块开发教程

## 一、概述
时序预测旨在通过分析历史数据中的模式、趋势和周期性等特征，来预测未来某一时间点或时间段内可能发生的值或状态。这有助于企业和组织做出更加精准的决策，优化资源配置，减少风险，并抓住潜在的市场机会。这些时序数据通常来自于实际应用场景中的各种传感器、经济活动、社会行为等。例如，股票价格、气温变化、网站访问量、销售数据等都是典型的时序数据。

## 二、支持模型列表

<details>
   <summary> 👉模型列表详情</summary>

|模型名称|mse|mae|模型存储大小（M)|介绍|
|-|-|-|-|-|
|DLinear|0.382|0.394|76k|DLinear结构简单，效率高且易用的时序预测模型|
|Nonstationary|0.600|0.515|60.3M|基于transformer结构，针对性优化非平稳时间序列的长时序预测模型|
|PatchTST|0.385|0.397|2.2M|PatchTST是兼顾局部模式和全局依赖关系的高精度长时序预测模型|
|TiDE|0.405|0.412|34.9M|TiDE是适用于处理多变量、长期的时间序列预测问题的高精度模型|
|TimesNet|0.417|0.431|5.2M|通过多周期分析，TimesNet是适应性强的高精度时间序列分析模型|

**注：以上精度指标测量自**[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)**测试数据集，输入序列长度为96，预测序列长度除 TiDE 外为96，TiDE为720 。**

</details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成是时序预测模块的推理，可以任意切换该模块下的模型，您也可以将时序预测的模块中的模型推理集成到您的项目中。

```bash
from paddlex import create_model
model = create_model("DLinear")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用PaddleX的二次开发能力，开发更好的时序预测模型。在使用PaddleX开发时序预测模型之前，请务必安装 PaddleTS 插件，安装过程可以参考[PaddleX本地安装教程](../../../installation/installation.md)。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，**只有通过数据校验的数据才可以进行模型训练**。此外，PaddleX为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX时序预测任务模块数据标注教程](../../../data_annotations/time_series_modules/time_series_forecasting.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ts_dataset_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括示例时序数据。

<details>
  <summary>👉 <b>校验结果详情（点击展开）</b></summary>


校验结果文件具体内容为：

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 12194,
    "train_table": [
      [
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      [
        "2016-07-01 00:00:00",
        5.827000141143799,
        2.009000062942505,
        1.5989999771118164,
        0.4620000123977661,
        4.203000068664552,
        1.3400000333786009,
        30.5310001373291
      ],
      [
        "2016-07-01 01:00:00",
        5.692999839782715,
        2.075999975204468,
        1.4919999837875366,
        0.4259999990463257,
        4.142000198364259,
        1.371000051498413,
        27.78700065612793
      ]
    ],
    "val_samples": 3484,
    "val_table": [
      [
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      [
        "2017-11-21 02:00:00",
        12.994000434875488,
        4.889999866485597,
        10.055999755859377,
        2.878000020980835,
        2.559000015258789,
        1.2489999532699585,
        4.7129998207092285
      ],
      [
        "2017-11-21 03:00:00",
        11.92199993133545,
        4.554999828338623,
        9.097000122070312,
        3.0920000076293945,
        2.559000015258789,
        1.2790000438690186,
        4.8540000915527335
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": "./dataset/ts_dataset_examples",
  "show_type": "csv",
  "dataset_type": "TSDataset"
}
```
上述校验结果中，`check_pass` 为 `True` 表示数据集格式符合要求，其他部分指标的说明如下：

* `attributes.train_samples`：该数据集训练集样本数量为 12194；
* `attributes.val_samples`：该数据集验证集样本数量为 3484；
* `attributes.train_table`：该数据集训练集样本示例数据前10行信息；
* `attributes.val_table`：该数据集训练集样本示例数据前10行信息；
**注**：只有通过数据校验的数据才可以训练和评估。

</details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过**修改配置文件**或是**追加超参数**的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details>
  <summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>


**（1）数据集格式转换**

时序预测支持 `xlsx 和 xlss` 格式的数据集转换为 `csv` 格式。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `convert`:
    * `enable`: 是否进行数据集格式转换，支持 `xlsx和xlss` 格式的数据集转换为 `CSV` 格式，默认为 `False`;
    * `src_dataset_type`: 如果进行数据集格式转换，无需设置源数据集格式，默认为 `null`，；
则需要修改配置如下：

```bash
......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: null
  ......
```
随后执行命令：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```
以上参数同样支持通过追加命令行参数的方式进行设置：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples \
    -o CheckDataset.convert.enable=True
```


**（2）数据集划分**

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `convert`:
    * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
    * `src_dataset_type`: 如果进行数据集格式转换，时序预测仅支持将xlsx标注文件转换为csv，无需设置源数据集格式，默认为 `null`；
  * `split`:
    * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
    * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证与 `val_percent` 的值之和为100；
    * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证与 `train_percent` 的值之和为100；
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
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```
数据划分执行之后，原有标注文件会被在原路径下重命名为 `xxx.bak`。

以上参数同样支持通过追加命令行参数的方式进行设置：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处高效率时序预测模型（DLinear）的训练为例：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`DLinear.yaml`）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅查阅模型对应任务模块的配置文件说明[PaddleX时序任务模型配置文件参数说明](../../instructions/config_parameters_time_series.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>



* 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段进行设置。
* PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。
* 训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)。
在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* `train_result.json`：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* `train.log`：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* `config.yaml`：训练配置文件，记录了本次训练的超参数的配置；
* `best_accuracy.pdparams.tar`、`scaler.pkl`、`.checkpoints` 、`.inference`：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；
</details>

### 4.3 模型评估
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`DLinear.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX时序任务模型配置文件参数说明](../../instructions/config_parameters_time_series.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>



在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=``./output/best_model/model.pdparams`。

在完成模型评估后，通常有以下产出：

在完成模型评估后，会产出`evaluate_result.json，其记录了`评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 mse 和 mae。

</details>

### 4.4 模型推理和模型集成
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`DLinear.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir=``"./output/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX时序任务模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到PaddleX产线中，也可以直接集成到您自己的项目中。

1.**产线集成**

时序预测模块可以集成的PaddleX产线有[时序预测](../../../pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.md)，只需要替换模型路径即可完成时序预测的模型更新。在产线集成中，你可以使用服务化部署来部署你得到的模型。

2.**模块集成**

您产出的权重可以直接集成到时序预测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。