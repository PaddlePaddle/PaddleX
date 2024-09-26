# PaddleX单模型开发工具使用教程
> 在使用单模型开发工具前，请先确保您按照[PaddleX安装教程](/docs_new/installation/installation.md)完成了PaddleX的安装。

在 PaddleX中，单模型指的是产线中针对某一特定任务模块（如图像分类、目标检测、语义分割、文本识别等）所设计的独立模型单元，这些模型都配备了官方的预训练权重，用户可以直接通过命令行或Python API 进行快速推理，同时也支持用户使用私有数据进行微调。

PaddleX 为每个模型都配备了相应的 `.yaml`配置文件，这些文件用于详细设定模型开发的各项参数。用户只需调整配置文件或追加命令行参数，即可轻松实现模型的训练、评估、推理预测以及训练数据集的校验。

本教程以图像分类模型（`PP-LCNet_x1_0`）为例，介绍PaddleX单模型开发工具的使用方法。
## 1. 数据准备
在进行模型训练前，需要准备相应任务模块的数据集，PaddleX针对各个任务模块都提供了Demo数据集及详细的数据准备教程，可以查阅[PaddleX模型训练数据准备教程](/docs_new/data_annotations/data_prepare_tutorial.md)进行数据准备。
## 2. 模型训练
一条命令即可完成模型的训练，以此处图像分类模型（`PP-LCNet_x1_0`）的训练为例：
```python
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
需要如下几步：
* 指定模型的.yaml 配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅查阅模型配置文件说明：[PaddleX模型配置文件参数说明](/docs_new/customization/model_train/config_parameters_common.md)。

**注:**
* 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output`字段进行设置。
* PaddleX对您屏蔽了动态图权重和静态图权重的概念。除时序模型外，在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。
* 训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](/docs_new/support_list/models_list.md)。
  
在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* `train_result.json`：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* `train.log`：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* `config.yaml`：训练配置文件，记录了本次训练的超参数的配置；
* `.pdparams`、`.pdema`、`.pdopt.pdstate`、`.pdiparams`、`.pdmodel`：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；

## 3. 模型评估
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：
```python
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml `配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX模型配置文件参数说明](/docs_new/customization/model_train/config_parameters_common.md)。

**注：**
在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model.pdparams`。

在完成模型评估后，通常有以下产出：

## 4. 模型推理

在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测。在PaddleX中实现模型推理预测可以通过两种方式：命令行和wheel 包。
> 命令行推理
> 
通过命令行的方式进行推理预测，只需如下一条命令：
```python
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model" \
    -o Predict.input_path="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"
```

与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml `配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model"`
* 指定输入数据路径：`-o Predict.input_path="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX模型配置文件参数说明](/docs_new/customization/model_train/config_parameters_common.md)。

* 也可以用PaddleX 的 wheel 包进行推理，方便地将模型集成到您自己的项目中。集成方式如下：

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

关于单模型Python集成的更多介绍，您可以参考[PaddleX单模型Python脚本使用说明](/docs_new/quick_inference/model_inference_with_python.md)。