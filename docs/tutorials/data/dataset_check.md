# PaddleX 数据集校验

PaddleX 针对常见 AI 任务模块，给出通用简明的数据集规范，涵盖数据集名称、组织结构、标注格式。您可以参考下面不同任务的说明准备数据，进而可以通过 PaddleX 的数据校验，最后完成全流程任务开发。在数据校验过程中，PaddleX 支持额外的功能，如数据集格式转换、数据集划分等，您可以根据自己的需求选择使用。

## 1. 图像分类任务模块数据校验

### 1.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了图像分类 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
```

### 1.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。相关信息会保存在当前目录的 `./output/check_dataset` 目录下。

<!-- 这里需要增加输出的说明，以及产出的说明，畅达 -->

**注**：只有通过数据校验的数据才可以训练和评估。


### 1.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    <!-- 这里需要增加详细的说明，廷权 -->
    * `dst_dataset_name`: 生成的数据集目录名，PaddleX 在数据校验时，会产生一个新的数据集；

    * `convert`:
        * `enable`: 是否进行数据集格式转换；
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式；
    * `split`:
        * `enable`: 是否进行重新划分数据集；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比；

以上参数同样支持通过追加命令行参数的方式进行设置，如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split=True -o CheckDataset.train_percent=0.8 -o CheckDataset.val_percent=0.2`。


## 目标检测任务模块数据校验

## 语义分割任务模块数据校验

## 实例分割任务模块数据校验

## 文本检测任务模块数据校验

## 文本识别任务模块数据校验

## 表格识别任务模块数据校验

## 时序预测任务模块数据校验

## 时序异常检测任务模块数据校验

## 时序分类任务模块数据校验
