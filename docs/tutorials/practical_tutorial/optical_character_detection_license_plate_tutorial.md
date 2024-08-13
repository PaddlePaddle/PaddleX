# PaddleX 3.0 通用OCR模型产线———车牌识别教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考[ PaddleX 安装](../INSTALL.md)。此处以一个车牌识别的任务为例子，介绍模型产线工具的使用流程。

## 1. 选择产线

首先，需要根据您的任务场景，选择对应的 PaddleX 产线，此处为车牌识别，需要了解到这个任务属于文本检测任务，对应 PaddleX 的通用OCR产线。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的[模型产线列表](../pipelines/support_pipeline_list.md)中了解相关产线的能力介绍。


## 2. 快速体验

PaddleX 提供了两种体验的方式，一种是可以直接通过 PaddleX wheel 包在本地体验，另外一种是可以在 **AI Studio 星河社区**上体验。

  - 本地体验方式：
    ```bash
    paddlex --pipeline OCR \
        --model PP-OCRv4_server_det PP-OCRv4_server_rec \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg
    ```

  - 星河社区体验方式：前往[AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine)，点击【创建产线】，创建【**通用OCR**】产线进行快速体验；

  快速体验产出推理结果示例：
  <center>

  <img src="https://github.com/user-attachments/assets/513f93b8-6f21-41d7-a214-016b21aa93d5" width=600>

  </center>

当体验完该产线之后，需要确定产线是否符合预期（包含精度、速度等），产线包含的模型是否需要继续微调，如果模型的速度或者精度不符合预期，则需要根据模型选择选择可替换的模型继续测试，确定效果是否满意。如果最终效果均不满意，则需要微调模型。

## 3. 选择模型

PaddleX 提供了 2 个端到端的文本检测模型，具体可参考 [模型列表](../models/support_model_list.md)，其中模型的 benchmark 如下：

| 模型列表         | 检测Hmean(%) | 识别 Avg Accuracy(%) | GPU 推理耗时(ms) | CPU 推理耗时(ms) | 模型存储大小(M) |
| --------------- | ----------- | ------------------- | --------------- | --------------- |---------------|
| PP-OCRv4_server	| 82.69       | 79.20               | 	22.20346	    | 2662.158        | 	        198 |
| PP-OCRv4_mobile	| 77.79       | 78.20	              | 2.719474	      | 79.1097         | 	         15 |

**注：以上精度指标为 PaddleOCR 自建中文数据集验证集 检测Hmean 和 识别 Avg Accuracy，GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。**
简单来说，表格从上到下，模型推理速度更快，从下到上，模型精度更高。本教程以 `PP-OCRv4_server` 模型为例，完成一次模型全流程开发。你可以依据自己的实际使用场景，判断并选择一个合适的模型做训练，训练完成后可在产线内评估合适的模型权重，并最终用于实际使用场景中。

## 4. 数据准备和校验
### 4.1 数据准备

本教程采用 `车牌识别数据集` 作为示例数据集，可通过以下命令获取示例数据集。如果您使用自备的已标注数据集，需要按照 PaddleX 的格式要求对自备数据集进行调整，以满足 PaddleX 的数据格式要求。关于数据格式介绍，您可以参考 [PaddleX 数据格式介绍](../data/dataset_format.md)。如果您有一批待标注数据，可以参考 [通用OCR数据标注指南](../data/annotation/OCRAnnoTools.md) 完成数据标注。

数据集获取命令：
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ccpd_text_det.tar -P ./dataset
tar -xf ./dataset/ccpd_text_det.tar -C ./dataset/
```

### 4.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/text_detection/PP-OCRv4_server_det.yaml \
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
      "..\/..\/ccpd_text_det\/images\/0274305555556-90_266-204&460_520&548-516&548_209&547_204&464_520&460-0_0_3_25_24_24_24_26-63-89.jpg",
      "..\/..\/ccpd_text_det\/images\/0126171875-90_267-294&424_498&486-498&486_296&485_294&425_496&424-0_0_3_24_33_32_30_31-157-29.jpg",
      "..\/..\/ccpd_text_det\/images\/0371516927083-89_254-178&423_517&534-517&534_204&525_178&431_496&423-1_0_3_24_33_31_29_31-117-667.jpg",
      "..\/..\/ccpd_text_det\/images\/03349609375-90_268-211&469_526&576-526&567_214&576_211&473_520&469-0_0_3_27_31_32_29_32-174-48.jpg",
      "..\/..\/ccpd_text_det\/images\/0388454861111-90_269-138&409_496&518-496&518_138&517_139&410_491&409-0_0_3_24_27_26_26_30-174-148.jpg",
      "..\/..\/ccpd_text_det\/images\/0198741319444-89_112-208&517_449&600-423&593_208&600_233&517_449&518-0_0_3_24_28_26_26_26-87-268.jpg",
      "..\/..\/ccpd_text_det\/images\/3027782118055555555-91_92-186&493_532&574-529&574_199&565_186&497_532&493-0_0_3_27_26_30_33_32-73-336.jpg",
      "..\/..\/ccpd_text_det\/images\/034375-90_258-168&449_528&546-528&542_186&546_168&449_525&449-0_0_3_26_30_30_26_33-94-221.jpg",
      "..\/..\/ccpd_text_det\/images\/0286501736111-89_92-290&486_577&587-576&577_290&587_292&491_577&486-0_0_3_17_25_28_30_33-134-122.jpg",
      "..\/..\/ccpd_text_det\/images\/02001953125-92_103-212&486_458&569-458&569_224&555_212&486_446&494-0_0_3_24_24_25_24_24-88-24.jpg"
    ],
    "val_samples": 1001,
    "val_sample_paths": [
      "..\/..\/ccpd_text_det\/images\/3056141493055555554-88_93-205&455_603&597-603&575_207&597_205&468_595&455-0_0_3_24_32_27_31_33-90-213.jpg",
      "..\/..\/ccpd_text_det\/images\/0680295138889-88_94-120&474_581&623-577&605_126&623_120&483_581&474-0_0_5_24_31_24_24_24-116-518.jpg",
      "..\/..\/ccpd_text_det\/images\/0482421875-87_265-154&388_496&530-490&495_154&530_156&411_496&388-0_0_5_25_33_33_33_33-84-104.jpg",
      "..\/..\/ccpd_text_det\/images\/0347504340278-105_106-235&443_474&589-474&589_240&518_235&443_473&503-0_0_3_25_30_33_27_30-162-4.jpg",
      "..\/..\/ccpd_text_det\/images\/0205338541667-93_262-182&428_410&519-410&519_187&499_182&428_402&442-0_0_3_24_26_29_32_24-83-63.jpg",
      "..\/..\/ccpd_text_det\/images\/0380913628472-97_250-234&403_529&534-529&534_250&480_234&403_528&446-0_0_3_25_25_24_25_25-185-85.jpg",
      "..\/..\/ccpd_text_det\/images\/020598958333333334-93_267-256&471_482&563-478&563_256&546_262&471_482&484-0_0_3_26_24_25_32_24-102-115.jpg",
      "..\/..\/ccpd_text_det\/images\/3030323350694444445-86_131-170&495_484&593-434&569_170&593_226&511_484&495-11_0_5_30_30_31_33_24-118-59.jpg",
      "..\/..\/ccpd_text_det\/images\/3016158854166666667-86_97-243&471_462&546-462&527_245&546_243&479_453&471-0_0_3_24_30_27_24_29-98-40.jpg",
      "..\/..\/ccpd_text_det\/images\/0340831163194-89_264-177&412_488&523-477&506_177&523_185&420_488&412-0_0_3_24_30_29_31_31-109-46.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "\/mnt\/liujiaxuan01\/new\/new2\/ccpd_text_det",
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

<img src="https://github.com/user-attachments/assets/0b642f7d-437d-437d-8b20-c5806cd11308" width=600>

</center>

**注**：只有通过数据校验的数据才可以训练和评估。


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
python main.py -c paddlex/configs/text_detection/PP-OCRv4_server_det.yaml \
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

更多超参数介绍，请参考 [PaddleX 超参数介绍](../base/hyperparameters_introduction.md)。

**注：**
- 以上参数可以通过追加令行参数的形式进行设置，如指定模式为模型训练：`-o Global.mode=train`；指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。
- 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段
- PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。

**训练产出解释:**  

在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* train_result.json：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* train.log：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* config.yaml：训练配置文件，记录了本次训练的超参数的配置；
* .pdparams、.pdopt、.pdstates、.pdiparams、.pdmodel：模型权重相关文件，包括网络参数、优化器、静态图网络参数、静态图网络结构等；

### 5.2 模型评估

在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，只需一行命令：

```bash
python main.py -c paddlex/configs/text_detection/PP-OCRv4_server_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ccpd_text_det
```

与模型训练类似，模型评估支持修改配置文件或追加命令行参数的方式设置。

**注：** 在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model.pdparams`。

### 5.3 模型调优

在学习了模型训练和评估后，我们可以通过调整超参数来提升模型的精度。通过合理调整训练轮数，您可以控制模型的训练深度，避免过拟合或欠拟合；而学习率的设置则关乎模型收敛的速度和稳定性。因此，在优化模型性能时，务必审慎考虑这两个参数的取值，并根据实际情况进行灵活调整，以获得最佳的训练效果。

推荐在调试参数时遵循控制变量法：

1. 首先固定训练轮次为 10，批大小为 8，卡数选择 4 卡，总批大小是 32。
2. 基于 PP-OCRv4_server_det 模型启动四个实验，学习率分别为：0.00005，0.0001，0.0005，0.001。
3. 可以发现实验 4 精度最高的配置为学习率为0.001，同时观察验证集分数，精度在最后几轮仍在上涨。因此可以提升训练轮次为 20，模型精度会有进一步的提升。

学习率探寻实验结果：
<center>

| 实验ID | 学习率	 | 检测Hmean(%)|
|-----------|-----|-------|
|1	 | 0.00005 | 	99.06|
|2	 | 0.0001 | 	99.55|
|3	 | 0.0005	 | 99.60|
|4	 | 0.001	 | 99.70|
</center>

接下来，我们可以在学习率设置为 0.001 的基础上，增加训练轮次，对比下面实验 [4，5] 可知，训练轮次增大，模型精度有了进一步的提升。
<center>

| 实验ID | 	训练轮次	 | 检测 Hmean(%) |
|-----------|-----|-------|
| 4	 | 10 | 	99.70   |
| 5	 | 20	 | 99.80   |
</center>

**注：本教程为 4 卡教程，如果您只有 1 张 GPU，可通过调整训练卡数完成本次实验，但最终指标未必和上述指标对齐，属正常情况。**

## 6. 产线测试

将产线中的模型替换为微调后的模型进行测试，如：

```bash
paddlex --pipeline OCR \
        --model PP-OCRv4_server_det PP-OCRv4_server_rec \
        --model_dir output/best_accuracy None \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg
```

通过上述可在`./output`下生成预测结果，其中`case1.jpg`的预测结果如下：
<center>

<img src="https://github.com/user-attachments/assets/e16674f8-2beb-482c-8760-71fb06f0b51f" width="600"/>

</center>

## 7. 开发集成/部署
1. 此处提供了轻量级的 PaddleX Python API 的集成方式，使用 Python API 方式可以更加方便的将 PaddleX 产出模型集成到自己的项目中进行二次开发，详细集成方式可参考 [PaddleX 模型产线推理预测](../pipelines/pipeline_inference.md)。
此处提供轻量级的 PaddleX Python API 的集成方式，也提供高性能推理/服务化部署的方式部署模型。 PaddleX Python API 的集成方式如下：

```python
from paddlex import OCRPipeline
from paddlex import PaddleInferenceOption

text_det_model_name = "PP-OCRv4_server_det"
text_rec_model_name = "PP-OCRv4_server_rec"

text_det_model_dir = "./output/best_model_det"

pipeline = OCRPipeline(text_det_model_name=text_det_model_name, text_rec_model_name=text_rec_model_name, text_det_model_dir=text_det_model_dir, text_det_kernel_option=PaddleInferenceOption(), text_rec_kernel_option=PaddleInferenceOption())
result = pipeline.predict(
        {'input_path': "./dataset/ccpd_text_det/images/0243359375-92_266-236&396_503&488-499&488_236&486_244&396_503&407-0_0_3_24_26_32_29_33-109-97.jpg"}
    )

print(result["dt_polys"])
```  
2. PaddleX也提供了基于 FastDeploy 的高性能推理/服务化部署的方式进行模型部署。该部署方案支持更多的推理后端，并且提供高性能推理和服务化部署两种部署方式，能够满足更多场景的需求，具体流程可参考 [基于 FastDeploy 的模型产线部署]((../pipelines/pipeline_deployment_with_fastdeploy.md))。高性能推理和服务化部署两种部署方式的特点如下：
    * 高性能推理：运行脚本执行推理，或在程序中调用 Python/C++ 的推理 API。旨在实现测试样本的高效输入与模型预测结果的快速获取，特别适用于大规模批量刷库的场景，显著提升数据处理效率。
    * 服务化部署：采用 C/S 架构，以服务形式提供推理能力，客户端可以通过网络请求访问服务，以获取推理结果。
* PaddleX 高性能离线部署和服务化部署流程如下：

    1. 获取离线部署包。
        1. 在 [AIStudio 星河社区](https://aistudio.baidu.com/pipeline/mine) 根据本地训练模型产线创建对应产线，在“选择产线”页面点击“直接部署”。
        2. 在“产线部署”页面选择“导出离线部署包”，使用默认的模型方案，选择与本地测试环境对应的部署包运行环境，点击“导出部署包”。
        3. 待部署包导出完毕后，点击“下载离线部署包”，将部署包下载到本地。
        4. 点击“生成部署包序列号”，根据页面提示完成设备指纹的获取以及设备指纹与序列号的绑定，确保序列号对应的激活状态为“已激活“。
    2. 使用自训练模型替换离线部署包 `model` 目录中的模型。
    3. 根据需要选择要使用的部署SDK：`offline_sdk` 目录对应高性能推理SDK，`serving_sdk` 目录对应服务化部署SDK。按照SDK文档（README.md）中的说明，完成部署环境准备，建议使用文档提供的官方docker进行环境部署。
    4. 对于高性能推理方式部署，修改 `offline_sdk/python_example/fd_model_config.yaml` 中的 "model_path_root" 字段值为自训练模型存放目录，并使用如下命令完成模型高性能推理：

```bash
python infer.py --resource_path . --device gpu --serial_num <serial_number> --update_license True --backend paddle_option  --input_data_path https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg --is_visualize True
```

其他产线的 Python API 集成方式可以参考[PaddleX 模型产线推理预测](../pipelines/pipeline_inference.md)。
PaddleX 同样提供了高性能的离线部署和服务化部署方式，具体参考[基于 FastDeploy 的模型产线部署](../pipelines/pipeline_deployment_with_fastdeploy.md)。