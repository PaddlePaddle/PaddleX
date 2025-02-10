---
comments: true
---

# PaddleX 3.0 时序分类模型产线———心跳监测时序数据分类应用教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考 [PaddleX本地安装教程](../installation/installation.md)。此处以一个心跳时序数据的分类的任务为例子，介绍模型产线工具的使用流程。

## 1. 选择产线
首先，需要根据您的任务场景，选择对应的 PaddleX 产线，本任务目标就是基于心跳监测数据对时序分类模型进行训练，实现对心跳时间序列状况的分类。了解到这个任务属于时序分类任务，对应 PaddleX 的时序分类产线。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的 [PaddleX产线列表(CPU/GPU)](../support_list/pipelines_list.md) 中了解相关产线的能力介绍。

## 2. 快速体验
PaddleX 提供了两种体验的方式，一种是可以直接通过 PaddleX 在本地体验，另外一种是可以在 <b>AI Studio 星河社区</b>上体验。

* 本地体验方式：
```python
from paddlex import create_model
model = create_model("TimesNet_cls")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
* 星河社区体验方式：可以进入 [官方时序分类应用](https://aistudio.baidu.com/community/app/105707/webUI?source=appCenter) 体验时序分类任务的能力。
注：由于时序数据和场景紧密相关，时序任务的在线体验官方内置模型仅是在一个特定场景下的模型方案，并非通用方案，不适用其他场景，因此体验方式不支持使用任意的文件来体验官方模型方案效果。但是，在完成自己场景数据下的模型训练之后，可以选择自己训练的模型方案，并使用对应场景的数据进行在线体验。

## 3. 选择模型
PaddleX 提供了1个端到端的时序分类模型，具体可参考 [模型列表](../support_list/models_list.md)，其中模型的benchmark如下：

<table>
<thead>
<tr>
<th>模型名称</th>
<th>acc(%)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>TimesNet_cls</td>
<td>87.5</td>
<td>792K</td>
<td>通过多周期分析，TimesNet是适应性强的高精度时序分类模型</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 </b>UWaveGestureLibrary<b>。</b>

## 4. 数据准备和校验
### 4.1 数据准备
为了演示时序分类任务整个流程，我们将使用公开的 [Heartbeat 数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_classify_examples.tar) 进行模型训练及验证。Heartbeat 数据集是 UEA 时间序列分类数据集的场景之一，涉及到心跳监测医学诊断这一实际任务。该数据集由多组时间序列组成，每个数据点由标签变量、群组编号和 61 个特征变量组成。该数据集通常用于测试和验证时间分类预测模型的性能。

我们已经将该数据集转化为标准数据格式，可通过以下命令获取示例数据集。关于数据格式介绍，您可以参考 [时序分类模块开发教程](../module_usage/tutorials/time_series_modules/time_series_classification.md)。

数据集获取命令：

```
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_classify_examples.tar -P ./dataset
tar -xf ./dataset/ts_classify_examples.tar -C ./dataset/
```
* <b>数据注意事项</b>
  * 基于收集的真实数据，明确时序数据的分类目标，并定义相应的分类标签。例如，在股票价格分类中，标签可能是“上涨”或“下跌”。对于在一段时间是“上涨”的时间序列，可以作为一个样本（group），即这段时间序列每个时间点都具有共同的 group_id, 标签列我们都定义为“上涨”标签；对于在一段时间是“下跌”的时间序列，可以作为一个样本（group），即这段时间序列每个时间点都具有共同的 group_id, 标签列我们都定义为“下跌”标签。每一个 group，就是一个分类样本。
  * 时间频率一致：确保所有数据序列的时间频率一致，如每小时、每日或每周，对于不一致的时间序列，可以通过重采样方法调整到统一的时间频率。
  * 时间序列长度一致：确保每一个group的时间序列的长度一致。
  * 缺失值处理：为了保证数据的质量和完整性，可以基于专家经验或统计方法进行缺失值填充。
  * 非重复性：保证数据是按照时间顺序按行收集的，同一个时间点不能重复出现。
### 4.2 数据集校验
在对数据集校验时，只需一行命令：

```
python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在 log 中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括示例数据行。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 82620,
    "train_table": [
      [ ...
    ],
    ],
    "val_samples": 83025,
    "val_table": [
      [ ...
      ]
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/ts_classify_examples",
  "show_type": "csv",
  "dataset_type": "TSCLSDataset"
}
```
上述校验结果中数据部分已省略，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

* attributes.train_samples：该数据集训练集样本数量为 82620
* attributes.val_samples：该数据集验证集样本数量为 83025
* attributes.train_table：该数据集训练集样本示例数据行；
* attributes.val_table：该数据集验证集样本示例数据行；
<b>注</b>：只有通过数据校验的数据才可以训练和评估。

### 4.3 数据集格式转换/数据集划分（非必选）
如需对数据集格式进行转换或是重新划分数据集，可参考 [时序分类模块开发教程](../module_usage/tutorials/time_series_modules/time_series_classification.md)中的4.1.3。

## 5. 模型训练和评估
### 5.1 模型训练
在训练之前，请确保您已经对数据集进行了校验。完成 PaddleX 模型的训练，只需如下一条命令：

```
    python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_classify_examples \
    -o Train.epochs_iters=5 \
    -o Train.batch_size=16 \
    -o Train.learning_rate=0.0001 \
    -o Train.time_col=time \
    -o Train.target_cols=dim_0,dim_1,dim_2 \
    -o Train.freq=1 \
    -o Train.group_id=group_id \
    -o Train.static_cov_cols=label
```
在 PaddleX 中模型训练支持：修改训练超参数、单机单卡训练(时序模型仅支持单卡训练)等功能，只需修改配置文件或追加命令行参数。

PaddleX 中每个模型都提供了模型开发的配置文件，用于设置相关参数。模型训练相关的参数可以通过修改配置文件中 `Train` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `Global`：
  * `mode`：模式，支持数据校验（`check_dataset`）、模型训练（`train`）、模型评估（`evaluate`）、单例测试（`predict`）；
  * `device`：训练设备，可选`cpu`、`gpu`；可在该路径下[模型支持列表](../support_list/models_list.md)的文档中，查看不同设备上支持的模型；
* `Train`：训练超参数设置；
  * `epochs_iters`：训练轮次数设置；
  * `learning_rate`：训练学习率设置；
  * `batch_size`：训练单卡批大小设置；
  * `time_col`: 时间列，须结合自己的数据设置时间序列数据集的时间列的列名称；
  * `target_cols`:结合自己的数据，设置时间序列数据集的目标变量的列名称，可以为多个，多个之间用','分隔。一般目标变量和希望预测的目标越相关，模型精度通常越高。本教程中心跳监控数据集中的时间列名有 61 个特征变量，如：dim_0, dim_1 等；
  * `freq`：频率，须结合自己的数据设置时间频率，如：1min、5min、1h；
  * `group_id`：一个群组编号表示的是一个时序样本，相同编号的时序序列组成一个样本。结合自己的数据设置指定群组编号的列名称, 如：group_id；
  * `static_cov_cols`：代表时序的类别编号列，同一个样本的标签相同。结合自己的数据设置类别的列名称，如：label。
更多超参数介绍，请参考 [PaddleX时序任务模型配置文件参数说明](../module_usage/instructions/config_parameters_time_series.md)。

<b>注：</b>

* 以上参数可以通过追加令行参数的形式进行设置，如指定模式为模型训练：`-o Global.mode=train`；指定前 1 卡 gpu 训练：`-o Global.device=gpu:0`；设置训练轮次数为 10：`-o Train.epochs_iters=10`；
* 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段。

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
    python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_classify_examples \
    -o Evaluate.weight_path=./output/best_model/model.pdparams
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`TimesNet_cls.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX时序任务模型配置文件参数说明](../module_usage/instructions/config_parameters_time_series.md)。

<b>注：</b> 在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model/model.pdparams`。



### 5.3 模型调优
通过合理调整训练轮数，您可以控制模型的训练深度，避免过拟合或欠拟合；而学习率的设置则关乎模型收敛的速度和稳定性。因此，在优化模型性能时，务必审慎考虑这两个参数的取值，并根据实际情况进行灵活调整，以获得最佳的训练效果。
基于控制变量的方法，我们可以采用在初始阶段基于固定的较小轮次，多次调整学习率，从而找到较优学习率；之后再次增大训练轮次，进一步提升效果。下面我们详细介绍时序分类的调参方法:

推荐在调试参数时遵循控制变量法：

1. 首先固定训练轮次为 5，批大小为 16。
2. 基于 TimesNet_cls 模型启动三个实验，学习率分别为：0.00001，0.0001，0.001。
3. 可以发现实验三精度最高。因此固定学习率为 0.001，尝试增大训练轮次到 30。注：由于时序任务内置的 earlystop 机制，验证集精度在 10 个 patience epoch（耐心轮次）后没有提升时，会自动停止训练。如果需要更改早停训练的耐心轮次，可以手动修改配置文件中的超参数 patience 的值。
学习率探寻实验结果：



<table>
<thead>
<tr>
<th>实验</th>
<th>轮次</th>
<th>学习率</th>
<th>batch_size</th>
<th>训练环境</th>
<th>验证集准确率</th>
</tr>
</thead>
<tbody>
<tr>
<td>实验一</td>
<td>5</td>
<td>0.00001</td>
<td>16</td>
<td>1卡</td>
<td>72.20%</td>
</tr>
<tr>
<td>实验二</td>
<td>5</td>
<td>0.0001</td>
<td>16</td>
<td>1卡</td>
<td>72.20%</td>
</tr>
<tr>
<td>实验三</td>
<td>5</td>
<td>0.001</td>
<td>16</td>
<td>1卡</td>
<td>73.20%</td>
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
<th>训练环境</th>
<th>验证集准确率</th>
</tr>
</thead>
<tbody>
<tr>
<td>实验三</td>
<td>5</td>
<td>0.001</td>
<td>16</td>
<td>1卡</td>
<td>73.20%</td>
</tr>
<tr>
<td>实验四</td>
<td>30</td>
<td>0.001</td>
<td>16</td>
<td>1卡</td>
<td>75.10%</td>
</tr>
</tbody>
</table>

## 6. 产线测试
将模型目录设置为训练完成的模型进行测试，使用[测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_classification/test.csv)，进行预测：

```
python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_classification/test.csv"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`TimesNet_cls.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX时序任务模型配置文件参数说明](../module_usage/instructions/config_parameters_time_series.md)。

## 7. 开发集成/部署
如果通用时序分类产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

1. 若您需要使用微调后的模型权重，可以获取 ts_classification 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config ts_classification --save_path ./my_path
```

将微调后模型权重的本地路径填写至产线配置文件中的 `model_dir` 即可, 若您需要将通用时序分类产线直接应用在您的 Python 项目中，可以参考 如下示例：

```yaml
pipeline_name: ts_classification

SubModules:
  TSClassification:
    module_name: ts_classification
    model_name: TimesNet_cls
    model_dir: ./output/inference  # 此处替换为您训练后得到的模型权重本地路径
    batch_size: 1
```
随后，在您的 Python 代码中，您可以这样使用产线：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/ts_classification.yaml")
output = pipeline.predict("pre_ts.csv")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_csv("./output/") # 保存csv格式结果
```
更多参数请参考[时序分类产线使用教程](../pipeline_usage/tutorials/time_series_pipelines/time_series_classification.md)

2. 此外，PaddleX 时序异常检测产线也提供了服务化部署方式，详细说明如下：
* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX 服务化部署指南](../pipeline_deploy/serving.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。
