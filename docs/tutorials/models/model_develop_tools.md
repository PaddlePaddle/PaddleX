# PaddleX 单模型开发工具

PaddleX 提供了丰富的单模型，其是完成某一类任务的子模块的最小单元，模型开发完后，可以方便地集成到各类系统中。PaddleX 中的每个模型提供了官方权重，支持通过命令行方式直接推理预测和调用 Python API 预测。命令行方式直接推理预测可以快速体验模型推理效果，而 Python API 预测可以方便地集成到自己的项目中进行预测。在使用单模型开发工具之前，首先需要安装 PaddleX 的 wheel 包，安装方式请参考 [PaddleX 安装文档](../INSTALL.md)。

## 1. 模型训练

在训练之前，请确保您的数据集已经经过了[数据校验](../data/README.md)。经过数据校验的数据集才可以进行训练。PaddleX 提供了很多不同的任务模块，不同的模块下又内置了很多被广泛验证的高精度、高效率、精度效率均衡的模型。训练模型时，您只需要一行命令，即可发起相应任务的训练。本文档提供了图像分类任务模块的 `PP-LCNet_x1_0` 模型的训练和评估示例，其他任务模块的训练与图像分类类似。当您按照 [PaddleX 数据集标注](../data/annotation/README.md)和 [PaddleX 数据集校验](../data/dataset_check.md)准备好训练数据后，即可参考本文档完成所有 PaddleX 支持的模型训练。

完成 PaddleX 模型的训练，只需如下一条命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
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
- 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段进行设置。
- 在 OCR 和语义分割任务模块中，参数 `epochs_iters` 对应训练 Step 数，在其他任务模块中，参数 `epochs_iters` 对应训练 Epoch 数。

## 2. 模型评估

在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，只需一行命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```

与模型训练类似，模型评估支持修改配置文件或追加命令行参数的方式设置。

**注：** 在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model.pdparams`。

## 3. 模型推理

在完成后，即可使用训练好的模型权重进行推理预测。使用 PaddleX 模型，通过命令行的方式进行推理预测，只需如下一条命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model" \
    -o Predict.input_path="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"
```

**注：** 
 1. 对于 LaTeX_OCR_rec 模型，您需要将Predict.model_dir设置为"./output/best_accuracy"。
 
 2. PaddleX 允许使用 wheel 包进行模型集成，在此处，当您验证好自己的模型之后，即可使用 PaddleX 的 wheel 包进行推理，方便地将模型集成到您自己的项目中。集成方式如下：


```python
from paddlex import PaddleInferenceOption, create_model

model_name = "PP-LCNet_x1_0"

# 实例化 PaddleInferenceOption 设置推理配置
kernel_option = PaddleInferenceOption()
kernel_option.set_device("gpu:0")

# 调用 create_model 函数实例化预测模型
model = create_model(model_name=model_name, model_dir="/output/best_model", kernel_option=kernel_option)

# 调用预测模型 model 的 predict 方法进行预测
result = model.predict({'input_path': "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"})
```
关于 Python API 的更多介绍，您可以参考 [PaddleX 模型推理 API](model_inference_api.md)。

## 4. 须知事项

### 4.1 训练须知事项

- 训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以详情[模型库](../models/support_model_list.md)。
- PaddleX 对您屏蔽了动态图权重和静态图权重的概念。除时序模型外，在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。

### 4.2 训练产出解释

在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

- `train_result.json`：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
- `train.log`：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
- `config.yaml`：训练配置文件，记录了本次训练的超参数的配置；
- `.pdparams`、`.pdema`、`.pdopt.pdstate`、`.pdiparams`、`.pdmodel`：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；
