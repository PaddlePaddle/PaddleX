---
comments: true
---

# PaddleX 3.0 时序异常检测模型产线———设备异常检测应用教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考[PaddleX本地安装教程](../installation/installation.md)。此处以一个设备节点的异常检测的任务为例子，介绍模型产线工具的使用流程。

## 1. 选择产线
首先，需要根据您的任务场景，选择对应的 PaddleX 产线，本任务旨在识别和标记出设备节点中的异常行为或异常状态，帮助企业和组织及时发现和解决应用服务器节点中的问题，提高系统的可靠性和可用性。了解到这个任务属于时序异常检测任务，对应 PaddleX 的时序异常检测产线。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的[PaddleX产线列表(CPU/GPU)](../support_list/pipelines_list.md)中了解相关产线的能力介绍。

## 2. 快速体验
PaddleX 提供了两种体验的方式，一种是可以直接通过 PaddleX 在本地体验，另外一种是可以在 <b>AI Studio 星河社区</b>上体验。

* 本地体验方式：

```python
from paddlex import create_model
model = create_model("PatchTST_ad")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
* 星河社区体验方式：可以进入 [官方时序异常检测应用](https://aistudio.baidu.com/community/app/105708/webUI?source=appCenter) 体验时序异常检测任务的能力。
注：由于时序数据和场景紧密相关，时序任务的在线体验官方内置模型仅是在一个特定场景下的模型方案，并非通用方案，不适用其他场景，因此体验方式不支持使用任意的文件来体验官方模型方案效果。但是，在完成自己场景数据下的模型训练之后，可以选择自己训练的模型方案，并使用对应场景的数据进行在线体验。

## 3. 选择模型
PaddleX 提供了4个端到端的时序异常检测模型，具体可参考 [模型列表](../support_list/models_list.md)，其中模型的benchmark如下：

<table>
<thead>
<tr>
<th>模型名称</th>
<th>precison</th>
<th>recall</th>
<th>f1_score</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>DLinear_ad</td>
<td>0.9898</td>
<td>0.9396</td>
<td>0.9641</td>
<td>72.8K</td>
<td>DLinear_ad结构简单，效率高且易用的时序异常检测模型</td>
</tr>
<tr>
<td>Nonstationary_ad</td>
<td>0.9855</td>
<td>0.8895</td>
<td>0.9351</td>
<td>1.5MB</td>
<td>基于transformer结构，针对性优化非平稳时间序列的异常检测模型</td>
</tr>
<tr>
<td>AutoEncoder_ad</td>
<td>0.9936</td>
<td>0.8436</td>
<td>0.9125</td>
<td>32K</td>
<td>AutoEncoder_ad是经典的自编码结构的效率高且易用的时序异常检测模型</td>
</tr>
<tr>
<td>PatchTST_ad</td>
<td>0.9878</td>
<td>0.9070</td>
<td>0.9457</td>
<td>164K</td>
<td>PatchTST是兼顾局部模式和全局依赖关系的高精度时序异常检测模型</td>
</tr>
</tbody>
</table>
> <b>注：以上精度指标测量自</b>[PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)<b>数据集，时序长度为100。</b>

## 4. 数据准备和校验
### 4.1 数据准备
为了演示时序异常检测任务整个流程，我们将使用公开的 MSL 数据集进行模型训练及验证。PSM（火星科学实验室）数据集由来自美国国家航空航天局，具有 55 个维度，其中包含来自航天器监测系统的意外事件异常（ISA）报告的遥测异常数据。具有实际应用背景，能够更好地反映真实场景中的异常情况，通常用于测试和验证时间序列异常检测模型的性能。本教程中基于该数据集进行异常检测。

我们已经将该数据集转化为标准数据格式，可通过以下命令获取示例数据集。关于数据格式介绍，您可以参考 [时序异常检测模块开发教程](../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)。

数据集获取命令：

```
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/msl.tar -P ./dataset
tar -xf ./dataset/msl.tar -C ./dataset/
```
* <b>数据注意事项</b>
  * 时序异常检测是一个无监督学习任务，因此不需要标注训练数据。收集的训练样本尽可能保证都是正常数据，即没有异常，训练集的标签列均设置为 0，或者不设置标签列也是可以的。验证集为了验证精度，需要进行标注，对于在某个时间点是异常的点，该时间点的标签设置为 1，正常的时间点的标签为 0。
  * 缺失值处理：为了保证数据的质量和完整性，可以基于专家经验或统计方法进行缺失值填充。
  * 非重复性：保证数据是按照时间顺序按行收集的，同一个时间点不能重复出现。

### 4.2 数据集校验
在对数据集校验时，只需一行命令：

```
python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/msl
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在 log 中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括示例数据行。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 58317,
    "train_table": [
      [
        "timestamp",
        "0",
        "1",
        "2",
        "..."
      ]
      [
        "..."
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": "./dataset/msl",
  "show_type": "csv",
  "dataset_type": "TSADDataset"
}
```
上述校验结果中数据部分已省略，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

* attributes.train_samples：该数据集训练集样本数量为 58317
* attributes.val_samples：该数据集验证集样本数量为 73729
* attributes.train_table：该数据集训练集样本示例数据行；
* attributes.val_table：该数据集验证集样本示例数据行；
<b>注</b>：只有通过数据校验的数据才可以训练和评估。

### 4.3 数据集格式转换/数据集划分（非必选）
如需对数据集格式进行转换或是重新划分数据集，可参考[时序异常检测模块开发教程](../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)中的4.1.3。

## 5. 模型训练和评估
### 5.1 模型训练
在训练之前，请确保您已经对数据集进行了校验。完成 PaddleX 模型的训练，只需如下一条命令：

```
  python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/msl \
    -o Train.epochs_iters=5 \
    -o Train.batch_size=16 \
    -o Train.learning_rate=0.0001 \
    -o Train.time_col=timestamp \
    -o Train.feature_cols=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54 \
    -o Train.freq=1 \
    -o Train.label_col=label \
    -o Train.seq_len=96
```
在 PaddleX 中模型训练支持：修改训练超参数、单机单卡训练(时序模型仅支持单卡训练)等功能，只需修改配置文件或追加命令行参数。

PaddleX 中每个模型都提供了模型开发的配置文件，用于设置相关参数。模型训练相关的参数可以通过修改配置文件中 `Train` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `Global`：
  * `mode`：模式，支持数据校验（`check_dataset`）、模型训练（`train`）、模型评估（`evaluate`）、单例测试（`predict`）；
  * `device`：训练设备，可选`cpu`、`gpu`；可在 [PaddleX模型列表（CPU/GPU）](../support_list/models_list.md)同级目录的文档中，查看不同设备上支持的模型；
* `Train`：训练超参数设置；
  * `epochs_iters`：训练轮次数设置；
  * `learning_rate`：训练学习率设置；
  * `batch_size`：训练单卡批大小设置；
  * `time_col`: 时间列，须结合自己的数据设置时间序列数据集的时间列的列名称；
  * `feature_cols`:特征变量表示能够判断设备是否异常的相关变量，例如设备是否异常，可能与设备运转时的散热量有关。结合自己的数据，设置特征变量的列名称，可以为多个，多个之间用','分隔。本教程中设备监控数据集中的时间列名有 55 个特征变量，如：0, 1 等；
  * `freq`：频率，须结合自己的数据设置时间频率，如：1min、5min、1h；
  * `input_len`: 输入给模型的时间序列长度，会按照该长度对时间序列切片，预测该长度下这一段时序序列是否有异常；输入长度建议结合实际场景考虑。本教程中输入长度为 96。表示希望预测 96 个时间点是否有异常；
  * `label`：代表时序时间点是否异常的编号，异常点为 1，正常点为 0。本教程中异常监控数据集为 label。
更多超参数介绍，请参考 [PaddleX时序任务模型配置文件参数说明](../module_usage/instructions/config_parameters_time_series.md)。以上参数可以通过追加令行参数的形式进行设置，如指定模式为模型训练：`-o Global.mode=train`；指定前 1 卡 gpu 训练：`-o Global.device=gpu:0`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。



<details><summary> 更多说明（点击展开） </summary>

<ul>
<li>模型训练过程中，PaddleX 会自动保存模型权重文件，默认为<code>output</code>，如需指定保存路径，可通过配置文件中 <code>-o Global.output</code> 字段进行设置。</li>
<li>PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。</li>
<li>训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅<a href="../support_list/models_list.md">PaddleX模型列表（CPU/GPU）</a>。
在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</li>
</ul>
<p><b>训练产出解释:</b></p>
<p>在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</p>
<ul>
<li><code>train_result.json</code>：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；</li>
<li><code>train.log</code>：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；</li>
<li><code>config.yaml</code>：训练配置文件，记录了本次训练的超参数的配置；</li>
<li><code>best_accuracy.pdparams.tar</code>、<code>scaler.pkl</code>、<code>.checkpoints</code> 、<code>.inference*</code>：模型权重相关文件，包括网络参数、优化器、静态图权重等；</li>
</ul></details>

### 5.2 模型评估
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，只需一行命令：

```
    python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/msl
```
与模型训练类似，模型评估支持修改配置文件或追加命令行参数的方式设置。

<b>注：</b> 在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model/model.pdparams`。

在完成模型评估后，通常有以下产出：

在完成模型评估后，会产出`evaluate_result.json`，其记录了评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 f1、recall 和 precision。

### 5.3 模型调优
在学习了模型训练和评估后，我们可以通过调整超参数来提升模型的精度。通过合理调整训练轮数，您可以控制模型的训练深度，避免过拟合或欠拟合；而学习率的设置则关乎模型收敛的速度和稳定性。因此，在优化模型性能时，务必审慎考虑这两个参数的取值，并根据实际情况进行灵活调整，以获得最佳的训练效果。

基于控制变量的方法，我们可以采用在初始阶段基于固定的较小轮次，多次调整学习率，从而找到较优学习率；之后再次增大训练轮次，进一步提升效果。下面我们详细介绍时序异常检测的调参方法:

推荐在调试参数时遵循控制变量法：

1. 首先固定训练轮次为 5，批大小为 16，输入长度为 96。
2. 基于 PatchTST_ad 模型启动三个实验，学习率分别为：0.0001，0.0005，0.001。
3. 可以发现实验三精度最高，其配置为学习率为 0.001。因此学习率固定为 0.001，尝试增大训练轮次为 20。
4. 可以发现实验四与实验三的精度一致，说明无需再增大训练轮次数。
学习率探寻实验结果：

<table>
<thead>
<tr>
<th>实验</th>
<th>轮次</th>
<th>学习率</th>
<th>batch_size</th>
<th>输入长度</th>
<th>训练环境</th>
<th>验证集F1 score (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>实验一</td>
<td>5</td>
<td>0.0001</td>
<td>16</td>
<td>96</td>
<td>1卡</td>
<td>79.5</td>
</tr>
<tr>
<td>实验二</td>
<td>5</td>
<td>0.0005</td>
<td>16</td>
<td>96</td>
<td>1卡</td>
<td>80.1</td>
</tr>
<tr>
<td>实验三</td>
<td>5</td>
<td>0.001</td>
<td>16</td>
<td>96</td>
<td>1卡</td>
<td>80.9</td>
</tr>
</tbody>
</table>
增大训练轮次实验结果：

<table>
<thead>
<tr>
<th>实验</th>
<th>轮次</th>
<th>学习率</th>
<th>batch_size</th>
<th>输入长度</th>
<th>训练环境</th>
<th>验证集F1 score (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>实验三</td>
<td>5</td>
<td>0.0005</td>
<td>16</td>
<td>96</td>
<td>1卡</td>
<td>80.9</td>
</tr>
<tr>
<td>实验四</td>
<td>20</td>
<td>0.0005</td>
<td>16</td>
<td>96</td>
<td>1卡</td>
<td>80.9</td>
</tr>
</tbody>
</table>

## 6. 产线测试

将产线中的模型替换为微调后的模型进行测试，使用[测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv)进行预测：

```
python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PatchTST_ad.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX时序任务模型配置文件参数说明](../module_usage/instructions/config_parameters_time_series.md)。

## 7.开发集成/部署
如果通用时序异常检测产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

1. 若您需要使用微调后的模型权重，可以获取 ts_anomaly_detection 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config ts_anomaly_detection --save_path ./my_path
```

将微调后模型权重的本地路径填写至产线配置文件中的 `model_dir` 即可, 若您需要将通用时序分类产线直接应用在您的 Python 项目中，可以参考 如下示例：

```yaml
pipeline_name: ts_anomaly_detection

SubModules:
  TSAnomalyDetection:
    module_name: ts_anomaly_detection
    model_name: PatchTST_ad
    model_dir: ./output/inference  # 此处替换为您训练后得到的模型权重本地路径
    batch_size: 1
```

随后执行如下代码即可完成预测：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/ts_anomaly_detection.yaml")
output = pipeline.predict("pre_ts.csv")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_csv("./output/") # 保存csv格式结果
```

更多参数请参考[时序异常检测产线使用教程](../pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.md)

2. 此外，PaddleX 时序异常检测产线也提供了服务化部署方式，详细说明如下：
* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX 服务化部署指南](../pipeline_deploy/serving.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。
