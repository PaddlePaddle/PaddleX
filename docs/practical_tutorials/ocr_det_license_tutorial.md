---
comments: true
---

# PaddleX 3.0 通用OCR模型产线———车牌识别教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考[ PaddleX 安装](../installation/installation.md)。此处以一个车牌识别的任务为例子，介绍模型产线工具的使用流程。

## 1. 选择产线

首先，需要根据您的任务场景，选择对应的 PaddleX 产线，此处为车牌识别，需要了解到这个任务属于文本检测任务，对应 PaddleX 的通用OCR产线。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的[模型产线列表](../support_list/pipelines_list.md)中了解相关产线的能力介绍。


## 2. 快速体验

PaddleX 提供了两种体验的方式，一种是可以直接通过 PaddleX wheel 包在本地体验，另外一种是可以在 <b>AI Studio 星河社区</b>上体验。

  - 本地体验方式：
    ```bash
    paddlex --pipeline OCR \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg
    ```

  - 星河社区体验方式：前往[AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine)，点击【创建产线】，创建【<b>通用OCR</b>】产线进行快速体验；

  快速体验产出推理结果示例：
  <center>

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/ocr/01.png" width=600>

  </center>

当体验完该产线之后，需要确定产线是否符合预期（包含精度、速度等），产线包含的模型是否需要继续微调，如果模型的速度或者精度不符合预期，则需要根据模型选择选择可替换的模型继续测试，确定效果是否满意。如果最终效果均不满意，则需要微调模型。

## 3. 选择模型

PaddleX 提供了 4 个端到端的文本检测模型，具体可参考 [模型列表](../support_list/models_list.md)，其中模型的 benchmark 如下：

<table>
<thead>
<tr>
<th>模型</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.56</td>
<td>83.3501</td>
<td>2434.01</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.35</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.7</td>
<td>PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td>78.68</td>
<td>-</td>
<td>-</td>
<td>2.1</td>
<td>PP-OCRv3 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td>80.11</td>
<td>-</td>
<td>-</td>
<td>102.1</td>
<td>PP-OCRv3 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
</tbody>
</table>

## 4. 数据准备和校验
### 4.1 数据准备

本教程采用 `车牌识别数据集` 作为示例数据集，可通过以下命令获取示例数据集。如果您使用自备的已标注数据集，需要按照 PaddleX 的格式要求对自备数据集进行调整，以满足 PaddleX 的数据格式要求。关于数据格式介绍，您可以参考 [PaddleX 文本检测/文本识别任务模块数据标注教程](../data_annotations/ocr_modules/text_detection_recognition.md)。

数据集获取命令：
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ccpd_text_det.tar -P ./dataset
tar -xf ./dataset/ccpd_text_det.tar -C ./dataset/
```

### 4.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ccpd_text_det
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在 log 中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 5769,
    "train_sample_paths": [
      "check_dataset\/demo_img\/0274305555556-90_266-204&460_520&548-516&548_209&547_204&464_520&460-0_0_3_25_24_24_24_26-63-89.jpg",
      "check_dataset\/demo_img\/0126171875-90_267-294&424_498&486-498&486_296&485_294&425_496&424-0_0_3_24_33_32_30_31-157-29.jpg",
      "check_dataset\/demo_img\/0371516927083-89_254-178&423_517&534-517&534_204&525_178&431_496&423-1_0_3_24_33_31_29_31-117-667.jpg",
      "check_dataset\/demo_img\/03349609375-90_268-211&469_526&576-526&567_214&576_211&473_520&469-0_0_3_27_31_32_29_32-174-48.jpg",
      "check_dataset\/demo_img\/0388454861111-90_269-138&409_496&518-496&518_138&517_139&410_491&409-0_0_3_24_27_26_26_30-174-148.jpg",
      "check_dataset\/demo_img\/0198741319444-89_112-208&517_449&600-423&593_208&600_233&517_449&518-0_0_3_24_28_26_26_26-87-268.jpg",
      "check_dataset\/demo_img\/3027782118055555555-91_92-186&493_532&574-529&574_199&565_186&497_532&493-0_0_3_27_26_30_33_32-73-336.jpg",
      "check_dataset\/demo_img\/034375-90_258-168&449_528&546-528&542_186&546_168&449_525&449-0_0_3_26_30_30_26_33-94-221.jpg",
      "check_dataset\/demo_img\/0286501736111-89_92-290&486_577&587-576&577_290&587_292&491_577&486-0_0_3_17_25_28_30_33-134-122.jpg",
      "check_dataset\/demo_img\/02001953125-92_103-212&486_458&569-458&569_224&555_212&486_446&494-0_0_3_24_24_25_24_24-88-24.jpg"
    ],
    "val_samples": 1001,
    "val_sample_paths": [
      "check_dataset\/demo_img\/3056141493055555554-88_93-205&455_603&597-603&575_207&597_205&468_595&455-0_0_3_24_32_27_31_33-90-213.jpg",
      "check_dataset\/demo_img\/0680295138889-88_94-120&474_581&623-577&605_126&623_120&483_581&474-0_0_5_24_31_24_24_24-116-518.jpg",
      "check_dataset\/demo_img\/0482421875-87_265-154&388_496&530-490&495_154&530_156&411_496&388-0_0_5_25_33_33_33_33-84-104.jpg",
      "check_dataset\/demo_img\/0347504340278-105_106-235&443_474&589-474&589_240&518_235&443_473&503-0_0_3_25_30_33_27_30-162-4.jpg",
      "check_dataset\/demo_img\/0205338541667-93_262-182&428_410&519-410&519_187&499_182&428_402&442-0_0_3_24_26_29_32_24-83-63.jpg",
      "check_dataset\/demo_img\/0380913628472-97_250-234&403_529&534-529&534_250&480_234&403_528&446-0_0_3_25_25_24_25_25-185-85.jpg",
      "check_dataset\/demo_img\/020598958333333334-93_267-256&471_482&563-478&563_256&546_262&471_482&484-0_0_3_26_24_25_32_24-102-115.jpg",
      "check_dataset\/demo_img\/3030323350694444445-86_131-170&495_484&593-434&569_170&593_226&511_484&495-11_0_5_30_30_31_33_24-118-59.jpg",
      "check_dataset\/demo_img\/3016158854166666667-86_97-243&471_462&546-462&527_245&546_243&479_453&471-0_0_3_24_30_27_24_29-98-40.jpg",
      "check_dataset\/demo_img\/0340831163194-89_264-177&412_488&523-477&506_177&523_185&420_488&412-0_0_3_24_30_29_31_31-109-46.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "ccpd_text_det",
  "show_type": "image",
  "dataset_type": "TextDetDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.train_samples：该数据集训练集样本数量为 5769；
- attributes.val_samples：该数据集验证集样本数量为 1001；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/ocr/02.png" width=600>

</center>

<b>注</b>：只有通过数据校验的数据才可以训练和评估。


### 4.3 数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 `val_percent` 值加和为 100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为 0-100 之间的任意整数，需要保证和 `train_percent` 值加和为 100；

数据划分时，原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 5. 模型训练和评估
### 5.1 模型训练

在训练之前，请确保您已经对数据集进行了校验。完成 PaddleX 模型的训练，只需如下一条命令：

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ccpd_text_det
```

在 PaddleX 中模型训练支持：修改训练超参数、单机单卡/多卡训练等功能，只需修改配置文件或追加命令行参数。

PaddleX 中每个模型都提供了模型开发的配置文件，用于设置相关参数。模型训练相关的参数可以通过修改配置文件中 `Train` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `Global`：
    * `mode`：模式，支持数据校验（`check_dataset`）、模型训练（`train`）、模型评估（`evaluate`）；
    * `device`：训练设备，可选`cpu`、`gpu`、`xpu`、`npu`、`mlu`，除 cpu 外，多卡训练可指定卡号，如：`gpu:0,1,2,3`；
* `Train`：训练超参数设置；
    * `epochs_iters`：训练轮次数设置；
    * `learning_rate`：训练学习率设置；

更多超参数介绍，请参考 [PaddleX 通用模型配置文件参数说明](../module_usage/instructions/config_parameters_common.md)。

<b>注：</b>
- 以上参数可以通过追加令行参数的形式进行设置，如指定模式为模型训练：`-o Global.mode=train`；指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。
- 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段
- PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。

<b>训练产出解释:</b>

在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* train_result.json：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* train.log：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* config.yaml：训练配置文件，记录了本次训练的超参数的配置；
* .pdparams、.pdopt、.pdstates、.pdiparams、.pdmodel：模型权重相关文件，包括网络参数、优化器、静态图网络参数、静态图网络结构等；

### 5.2 模型评估

在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，只需一行命令：

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ccpd_text_det
```

与模型训练类似，模型评估支持修改配置文件或追加命令行参数的方式设置。

<b>注：</b> 在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams`。

### 5.3 模型调优

在学习了模型训练和评估后，我们可以通过调整超参数来提升模型的精度。通过合理调整训练轮数，您可以控制模型的训练深度，避免过拟合或欠拟合；而学习率的设置则关乎模型收敛的速度和稳定性。因此，在优化模型性能时，务必审慎考虑这两个参数的取值，并根据实际情况进行灵活调整，以获得最佳的训练效果。

推荐在调试参数时遵循控制变量法：

1. 首先固定训练轮次为 10，批大小为 8，卡数选择 4 卡，总批大小是 32。
2. 基于 PP-OCRv4_server_det 模型启动四个实验，学习率分别为：0.00005，0.0001，0.0005，0.001。
3. 可以发现实验 4 精度最高的配置为学习率为0.001，同时观察验证集分数，精度在最后几轮仍在上涨。因此可以提升训练轮次为 20，模型精度会有进一步的提升。

学习率探寻实验结果：
<center>

<table>
<thead>
<tr>
<th>实验ID</th>
<th>学习率</th>
<th>检测Hmean(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0.00005</td>
<td>99.06</td>
</tr>
<tr>
<td>2</td>
<td>0.0001</td>
<td>99.55</td>
</tr>
<tr>
<td>3</td>
<td>0.0005</td>
<td>99.60</td>
</tr>
<tr>
<td>4</td>
<td>0.001</td>
<td>99.70</td>
</tr>
</tbody>
</table>
</center>

接下来，我们可以在学习率设置为 0.001 的基础上，增加训练轮次，对比下面实验 [4，5] 可知，训练轮次增大，模型精度有了进一步的提升。
<center>

<table>
<thead>
<tr>
<th>实验ID</th>
<th>训练轮次</th>
<th>检测 Hmean(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>4</td>
<td>10</td>
<td>99.70</td>
</tr>
<tr>
<td>5</td>
<td>20</td>
<td>99.80</td>
</tr>
</tbody>
</table>
</center>

<b>注：本教程为 4 卡教程，如果您只有 1 张 GPU，可通过调整训练卡数完成本次实验，但最终指标未必和上述指标对齐，属正常情况。</b>

## 6. 产线测试

产线中的模型替换为微调后的模型进行测试，可以获取 OCR 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config OCR --save_path ./my_path
```

将配置文件中的`SubModules.TextDetection.model_dir`修改为自己的模型路径：

```yaml
SubModules:
  TextDetection:
    module_name: text_detection
    model_name: PP-OCRv4_mobile_det
    model_dir: output/best_accuracy/inference # 替换为微调后的文本检测模型权重路径
    ...
```

随后，基于Python脚本方式，加载修改后的产线配置文件即可:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="my_path/OCR.yaml")

output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")

```

通过上述可在`./output`下生成预测结果，其中`case1.jpg`的预测结果如下：
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/ocr/03.png" width="600"/>

</center>

## 7. 开发集成/部署
如果通用 OCR 产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。
1. 直接将训练好的模型应用在您的 Python 项目中，可以参考如下示例代码:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="paddlex/configs/pipelines/OCR.yaml")

output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")

```
更多参数请参考 [OCR 产线使用教程](../pipeline_usage/tutorials/ocr_pipelines/OCR.md)。

2. 此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

* 高性能部署：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考 [PaddleX 高性能推理指南](../pipeline_deploy/high_performance_inference.md)。
* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考 [PaddleX 服务化部署指南](../pipeline_deploy/serving.md)。
* 端侧部署：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考 [PaddleX端侧部署指南](../pipeline_deploy/edge_deploy.md)。

您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。
