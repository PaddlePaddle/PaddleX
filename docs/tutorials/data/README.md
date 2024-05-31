# 数据校验

PaddleX 支持对数据集进行校验，确保数据集内容符合 PaddleX 的相关要求。同时在数据校验时，能够对数据集进行分析，统计数据集的基本信息。此外，PaddleX 支持将其他常见数据集类型转换为 PaddleX 规定的数据集格式，以及对数据集按比例重新划分训练集、验证集。

## 1. 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml -o Global.mode=check_dataset
```

## 2. 数据集格式转换\数据集划分

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `dst_dataset_name`: 生成的数据集目录名，PaddleX 在数据校验时，会产生一个新的数据集；
    * `convert`:
        * `enable`: 是否进行数据集格式转换；
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式；
    * `split`:
        * `enable`: 是否进行重新划分数据集；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比；

以上参数同样支持通过追加命令行参数的方式进行设置，如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split=True -o CheckDataset.train_percent=True -o CheckDataset.val_percent=True`。
