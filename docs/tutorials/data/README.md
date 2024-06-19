# 数据校验

PaddleX 支持对数据集进行校验，确保数据集格式符合 PaddleX 的相关要求。同时在数据校验时，能够对数据集进行分析，统计数据集的基本信息。此外，PaddleX 支持将其他常见数据集类型转换为 PaddleX 规定的数据集格式，以及对数据集按比例重新划分训练集、验证集。本文档提供了图像分类的示例，其他任务的数据校验与图像分类类似。详情[PaddleX 数据校验](./dataset_check.md)。


## 1. 数据准备
您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了图像分类 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
```

## 2. 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。相关信息会保存在当前目录的 `./output/check_dataset` 目录下。

<!-- 这里需要增加输出的说明，以及产出的说明，畅达 -->

**注**：只有通过数据校验的数据才可以训练和评估。


## 3. 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    <!-- 这里需要增加详细的说明，比如图像分类这里不支持转换，也需要说明，畅达 -->
    * `dst_dataset_name`: 生成的数据集目录名，PaddleX 在数据校验时，会产生一个新的数据集；

    * `convert`:
        * `enable`: 是否进行数据集格式转换；
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式；
    * `split`:
        * `enable`: 是否进行重新划分数据集；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比；

以上参数同样支持通过追加命令行参数的方式进行设置，如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split=True -o CheckDataset.train_percent=0.8 -o CheckDataset.val_percent=0.2`。
