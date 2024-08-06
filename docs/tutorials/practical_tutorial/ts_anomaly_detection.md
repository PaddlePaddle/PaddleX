# PaddleX 3.0 时序异常检测模型产线———设备异常检测应用教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考[ PaddleX 安装](../INSTALL.md)。此处以一个设备节点的异常检测的任务为例子，介绍模型产线工具的使用流程。

## 1. 选择产线

首先，需要根据您的任务场景，选择对应的 PaddleX 产线，本任务该任务旨在识别和标记出设备节点中的异常行为或异常状态，帮助企业和组织及时发现和解决应用服务器节点中的问题，提高系统的可靠性和可用性。了解到这个任务属于时序异常检测任务，对应 PaddleX 的时序异常检测产线。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的[模型产线列表](../pipelines/support_pipeline_list.md)中了解相关产线的能力介绍。


## 2. 快速体验

PaddleX 提供了两种体验的方式，一种是可以直接通过 PaddleX 在本地体验，另外一种是可以在 **AI Studio 星河社区**上体验。

  - 本地体验方式：
    ```bash
    python main.py -c paddlex/configs/ts_anomaly_detection/PatchTST_ad.yaml \
        -o Global.mode=predict \
        -o Predict.model_dir=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/PatchTST_ad_infer.tar \
        -o Predict.input_path=https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv
    ```

  - 星河社区体验方式：可以进入 [官方时序异常检测应用](https://aistudio.baidu.com/community/app/105708/webUI?source=appCenter) 体验时序异常检测任务的能力。


注：由于时序数据和场景紧密相关，时序任务的在线体验官方内置模型仅是在一个特定场景下的模型方案，并非通用方案，不适用其他场景，因此体验方式不支持使用任意的文件来体验官方模型方案效果。但是，在完成自己场景数据下的模型训练之后，可以选择自己训练的模型方案，并使用对应场景的数据进行在线体验。


## 3. 选择模型

PaddleX 提供了5个端到端的时序异常检测模型，具体可参考 [模型列表](../models/support_model_list.md)，其中模型的benchmark如下：

<center>

| 模型列表          | precision | recall | f1_score | 模型存储大小 |
|:--------------|:---------:|:------:|:--------:|:---------:|
| DLinear_ad	      |  0.904	   | 0.891  |  0.897   |   0.9M    |
| Nonstationary_ad |  0.901	   | 0.938	 |  0.918   |  19.1MB   |
| AutoEncoder_ad	  |  0.897	   | 0.860	 |  0.876   |   0.4M    |
| PatchTST_ad	     |  0.900	   | 0.925	 |  0.913   |   2.1M    |
| TimesNet_ad	     |  0.899	   | 0.935	 |  0.917   |   5.4M    |

</center>

> **注：以上精度指标测量自 <a href="https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info">SWAT</a> 数据集，重建长度为 100。**

你可以依据自己的实际使用场景，判断并选择一个合适的模型做训练，训练完成后可在产线内评估合适的模型权重，并最终用于实际使用场景中。

## 4. 数据准备和校验
### 4.1 数据准备

为了演示时序异常检测任务整个流程，我们将使用公开的 MSL 数据集(url)进行模型训练及验证。PSM（火星科学实验室）数据集由来自美国国家航空航天局，具有 55 个维度，其中包含来自航天器监测系统的意外事件异常（ISA）报告的遥测异常数据。具有实际应用背景，能够更好地反映真实场景中的异常情况，通常用于测试和验证时间序列异常检测模型的性能。本教程中基于该数据集进行异常检测。

我们已经将该数据集转化为标准数据格式，可通过以下命令获取示例数据集。关于数据格式介绍，您可以参考 [PaddleX 数据格式介绍](../data/dataset_format.md)。

数据集获取命令：
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/msl.tar -P ./dataset
tar -xf ./dataset/msl.tar -C ./dataset/
```

- **数据注意事项**

  - 时序异常检测是一个无监督学习任务，因此不需要标注训练数据。收集的训练样本尽可能保证都是正常数据，即没有异常，训练集的标签列均设置为 0，或者不设置标签列也是可以的。验证集为了验证精度，需要进行标注，对于在某个时间点是异常的点，该时间点的标签设置为 1，正常的时间点的标签为 0。

  - 缺失值处理：为了保证数据的质量和完整性，可以基于专家经验或统计方法进行缺失值填充。

  - 非重复性：保证数据是安装时间顺序按行收集的，同一个时间点不能重复出现。

### 4.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/PatchTST_ad.yaml \
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

- attributes.train_samples：该数据集训练集样本数量为 58317
- attributes.val_samples：该数据集验证集样本数量为 73729
- attributes.train_table：该数据集训练集样本示例数据行；
- attributes.val_table：该数据集验证集样本示例数据行；


**注**：只有通过数据校验的数据才可以训练和评估。


### 4.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，数据可选源格式为 `xlsx` 和 `xls`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 `val_percent` 值加和为 100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为 0-100 之间的任意整数，需要保证和 `train_percent` 值加和为 100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 5. 模型训练和评估
### 5.1 模型训练

在训练之前，请确保您已经对数据集进行了校验。完成 PaddleX 模型的训练，只需如下一条命令：

```bash
    python main.py -c paddlex/configs/ts_anomaly_detection/PatchTST_ad.yaml \
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
    * `device`：训练设备，可选`cpu`、`gpu`；可在该路径下[模型支持列表](https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-beta/docs/tutorials/models)的文档中，查看不同设备上支持的模型；
* `Train`：训练超参数设置；
    * `epochs_iters`：训练轮次数设置；
    * `learning_rate`：训练学习率设置；
    * `batch_size`：训练单卡批大小设置；
    * `time_col`: 时间列，须结合自己的数据设置时间序列数据集的时间列的列名称;
    * `feature_cols`:特征变量表示能够判断设备是否异常的相关变量，例如设备是否异常，可能与设备运转时的散热量有关。结合自己的数据，设置特征变量的列名称，可以为多个，多个之间用','分隔。本教程中设备监控数据集中的时间列名有 55 个特征变量，如：0, 1 等;
    * `freq`：频率，须结合自己的数据设置时间频率，如：1min、5min、1h;
    * `input_len`: 输入给模型的时间序列长度，会按照该长度对时间序列切片，预测该长度下这一段时序序列是否有异常；输入长度建议结合实际场景考虑。本教程中输入长度为 96。表示希望预测 96 个时间点是否有异常。
    * `label`：代表时序时间点是否异常的编号，异常点为 1，正常点为 0。本教程中异常监控数据集为 label。


更多超参数介绍，请参考 [PaddleX 超参数介绍](../base/hyperparameters_introduction.md)。

**注：**
- 以上参数可以通过追加令行参数的形式进行设置，如指定模式为模型训练：`-o Global.mode=train`；指定前 1 卡 gpu 训练：`-o Global.device=gpu:0`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。
- 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段


**训练产出解释:**  

在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* train_result.json：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* train.log：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* config.yaml：训练配置文件，记录了本次训练的超参数的配置；
* .pdparams、.pkl、*model_meta、*checkpoint、best_accuracy.pdparams.tar模型权重相关文件，包括网络参数、优化器、归一化、网络参数、数据信息和最佳模型权重压缩包等；

### 5.2 模型评估

在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，只需一行命令：

```bash
    python main.py -c paddlex/configs/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/msl
```

与模型训练类似，模型评估支持修改配置文件或追加命令行参数的方式设置。

**注：** 在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model/model.pdparams`。

### 5.3 模型调优

在学习了模型训练和评估后，我们可以通过调整超参数来提升模型的精度。通过合理调整训练轮数，您可以控制模型的训练深度，避免过拟合或欠拟合；而学习率的设置则关乎模型收敛的速度和稳定性。因此，在优化模型性能时，务必审慎考虑这两个参数的取值，并根据实际情况进行灵活调整，以获得最佳的训练效果。

基于控制变量的方法，我们可以采用在初始阶段基于固定的较小轮次，多次调整学习率，从而找到较优学习率；之后再次增大训练轮次，进一步提升效果。下面我们详细介绍时序异常检测的调参方法:

推荐在调试参数时遵循控制变量法：

1. 首先固定训练轮次为 5，批大小为 16，输入长度为 96。
2. 基于 PatchTST_ad 模型启动三个实验，学习率分别为：0.0001，0.0005，0.001。
3. 可以发现实验三精度最高，其配置为学习率为 0.001。因此学习率固定为 0.001，尝试增大训练轮次为 20。
4. 可以发现实验四与实验三的精度一致，说明无需再增大训练轮次数。

学习率探寻实验结果：
<center>

| 实验   | 轮次 | 学习率   | batch_size | 输入长度 | 训练环境 | 验证集F1 score (%) |
|-------|----|-------|-------------|------|------|----------|
| 实验一 | 5 | 0.0001 | 16          | 96  | 1卡   | 79.5   |
| 实验二 | 5 | 0.0005 | 16          | 96  | 1卡   | 80.1   |
| 实验三 | 5 | 0.001  | 16          | 96  | 1卡   | **80.9**  |
</center>

增大训练轮次实验结果：
<center>

| 实验   | 轮次 | 学习率   | batch_size | 输入长度| 训练环境 | 验证集F1 score (%)|
|-------|----|-------|-------------|------|------|----------|
| 实验三 | 5 | 0.0005 | 16          | 96  | 1卡   | 80.9   |
| 实验四 | 20 | 0.0005 | 16          | 96  | 1卡   | 80.9   |

</center>


## 6. 产线测试

将产线中的模型替换为微调后的模型进行测试，使用[测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv)进行预测：

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="output/best_model/model.pdparams" \
    -o Predict.input_path=https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv
```

通过上述可在`./output`下生成预测结果，其中`test.csv`的预测结果保存在result.csv中。


## 7. 部署
 
PaddleX 针对时序分析模型提供了 本地推理部署/服务化部署的方式进行模型部署。目前时序部署方案为动态图部署，提供本地推理和服务化部署两种部署方式，能够满足更多场景的需求。本地部署和服务化部署两种部署方式的特点如下：

    * 本地部署：运行脚本执行推理，或在程序中调用 Python 的推理 API。旨在实现测试样本的高效输入与模型预测结果的快速获取，特别适用于大规模批量刷库的场景，显著提升数据处理效率。
    * 服务化部署：采用 C/S 架构，以服务形式提供推理能力，客户端可以通过网络请求访问服务，以获取推理结果。
* PaddleX 本地部署和服务化部署流程如下：

    1. 获取离线部署包。
        1. 进入 [AIStudio 星河社区](https://aistudio.baidu.com/pipeline/mine) 根据本地训练模型产线创建对应产线，在“选择产线”页面点击“直接部署”。
        2. 在“产线部署”页面选择“导出离线部署包”，使用默认的模型方案，选择与本地测试环境对应的部署包运行环境，点击“导出部署包”。
        3. 待部署包导出完毕后，点击“下载离线部署包”，将部署包下载到本地。
        4. 点击“生成部署包序列号”，根据页面提示完成设备指纹的获取以及设备指纹与序列号的绑定，确保序列号对应的激活状态为“已激活“。
    2. 将自训练模型 `best_accuracy.pdparams.tar` 放在离线部署包 `model/ts_anomaly_detection_module/` 目录中。
    3. 根据需要选择要使用的部署SDK：`offline_sdk` 目录对应推理SDK，`serving_sdk` 目录对应服务化部署SDK。按照SDK文档（README.md）中的说明，完成部署环境准备。
```bash
python infer.py \
            --csv_path test.csv \
            --device gpu \
            --save_dir ./output_infer \
            --checkpoints ./model/ts_anomaly_detection_module/best_accuracy.pdparams.tar \
            --visual True \
            --serial_num <serial_num> \
            --update_license True
```

其他产线的 Python API 集成方式可以参考[PaddleX 模型产线推理预测](../pipelines/pipeline_inference.md)。
PaddleX 同样提供了高性能的离线部署和服务化部署方式，具体参考[基于 FastDeploy 的模型产线部署](../pipelines/pipeline_deployment_with_fastdeploy.md)。
