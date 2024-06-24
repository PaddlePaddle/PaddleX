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

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "dataset/label.txt",
    "num_classes": 102,
    "train_samples": 1020,
    "train_sample_paths": [
      "check_dataset/demo_img/image_01904.jpg",
      "check_dataset/demo_img/image_06940.jpg"
    ],
    "val_samples": 1020,
    "val_sample_paths": [
      "check_dataset/demo_img/image_01937.jpg",
      "check_dataset/demo_img/image_06958.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/cls_flowers_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.num_classes：该数据集类别数为 102；
- attributes.train_samples：该数据集训练集样本数量为 1020；
- attributes.val_samples：该数据集验证集样本数量为 1020；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
![样本分布直方图](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/open_source/quick_start/histogram.png)

**注**：只有通过数据校验的数据才可以训练和评估。


### 1.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，图像分类不支持格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，图像分类不支持数据转换，默认为 `null`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。


## 2.目标检测任务模块数据校验

### 2.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了目标检测 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```

### 2.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 3,
    "train_samples": 56,
    "train_sample_paths": [
      "check_dataset/demo_img/304.png",
      "check_dataset/demo_img/322.png"
    ],
    "val_samples": 14,
    "val_sample_paths": [
      "check_dataset/demo_img/114.png",
      "check_dataset/demo_img/206.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/det_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}  
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.num_classes：该数据集类别数为 3；
- attributes.train_samples：该数据集训练集样本数量为 56；
- attributes.val_samples：该数据集验证集样本数量为 14；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
![样本分布直方图](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/open_source/tutorials/data/dataset_check/object_detection/histogram.png)

**注**：只有通过数据校验的数据才可以训练和评估。


### 2.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，数据可选源格式为 `LabelMe`、`LabelMeWithUnlabeled`、`VOC` 和 `VOCWithUnlabeled`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 3.语义分割任务模块数据校验

### 3.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了语义分割 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar -P ./dataset
tar -xf ./dataset/seg_optic_examples.tar -C ./dataset/
```

### 3.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_optic_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_sample_paths": [
      "check_dataset/demo_img/P0005.jpg",
      "check_dataset/demo_img/P0050.jpg"
    ],
    "train_samples": 267,
    "val_sample_paths": [
      "check_dataset/demo_img/N0139.jpg",
      "check_dataset/demo_img/P0137.jpg"
    ],
    "val_samples": 76,
    "num_classes": 2
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/seg_optic_examples",
  "show_type": "image",
  "dataset_type": "SegDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.num_classes：该数据集类别数为 2；
- attributes.train_samples：该数据集训练集样本数量为 267；
- attributes.val_samples：该数据集验证集样本数量为 76；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
![样本分布直方图](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/open_source/tutorials/data/dataset_check/semantic_segmentation/histogram.png)

**注**：只有通过数据校验的数据才可以训练和评估。


### 3.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，数据可选源格式为 `LabelMe`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 4. 实例分割任务模块数据校验

### 4.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了实例分割 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/instance_seg_coco_examples.tar -P ./dataset
tar -xf ./dataset/instance_seg_coco_examples.tar -C ./dataset/
```

### 4.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/instance_seg_coco_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 2,
    "train_samples": 79,
    "train_sample_paths": [
      "check_dataset/demo_img/pexels-photo-634007.jpeg",
      "check_dataset/demo_img/pexels-photo-59576.png"
    ],
    "val_samples": 19,
    "val_sample_paths": [
      "check_dataset/demo_img/peasant-farmer-farmer-romania-botiza-47862.jpeg",
      "check_dataset/demo_img/pexels-photo-715546.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/instance_seg_coco_examples",
  "show_type": "image",
  "dataset_type": "COCOInstSegDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.num_classes：该数据集类别数为 2；
- attributes.train_samples：该数据集训练集样本数量为 79；
- attributes.val_samples：该数据集验证集样本数量为 19；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
![样本分布直方图](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/open_source/tutorials/data/dataset_check/instance_segmentation/histogram.png)

**注**：只有通过数据校验的数据才可以训练和评估。


### 4.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，数据可选源格式为 `LabelMe`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 5. 文本检测任务模块数据校验

### 5.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了文本检测 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_det_dataset_examples.tar -C ./dataset/
```

### 5.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 200,
    "train_sample_paths": [
      "../dataset/ocr_det_dataset_examples/images/train_img_61.jpg",
      "../dataset/ocr_det_dataset_examples/images/train_img_289.jpg"
    ],
    "val_samples": 50,
    "val_sample_paths": [
      "../dataset/ocr_det_dataset_examples/images/val_img_61.jpg",
      "../dataset/ocr_det_dataset_examples/images/val_img_137.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/ocr_det_dataset_examples",
  "show_type": "image",
  "dataset_type": "TextDetDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.train_samples：该数据集训练集样本数量为 200；
- attributes.val_samples：该数据集验证集样本数量为 50；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
![样本分布直方图](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/open_source/tutorials/data/dataset_check/text_detection/histogram.png)

**注**：只有通过数据校验的数据才可以训练和评估。


### 4.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，文本检测不支持格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，文本检测不支持格式转换，默认为 `null`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 6. 文本识别任务模块数据校验

### 6.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了文本识别 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_rec_dataset_examples.tar -C ./dataset/
```

### 6.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 4468,
    "train_sample_paths": [
      "../dataset/ocr_rec_dataset_examples/images/train_word_1.png",
      "../dataset/ocr_rec_dataset_examples/images/train_word_10.png"
    ],
    "val_samples": 2077,
    "val_sample_paths": [
      "../dataset/ocr_rec_dataset_examples/images/val_word_1.png",
      "../dataset/ocr_rec_dataset_examples/images/val_word_10.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/ocr_rec_dataset_examples",
  "show_type": "image",
  "dataset_type": "MSTextRecDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.train_samples：该数据集训练集样本数量为 4468；
- attributes.val_samples：该数据集验证集样本数量为 2077；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
![样本分布直方图](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/open_source/tutorials/data/dataset_check/text_recognition/histogram.png)

**注**：只有通过数据校验的数据才可以训练和评估。


### 6.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，文本识别不支持格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，文本识别不支持格式转换，默认为 `null`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 7. 表格识别任务模块数据校验

### 7.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了表格识别 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar -P ./dataset
tar -xf ./dataset/table_rec_dataset_examples.tar -C ./dataset/
```

### 7.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/table_recognition/SLANet.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/table_rec_dataset_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 2000,
    "train_sample_paths": [
      "../dataset/table_rec_dataset_examples/images/border_right_7384_X9UFEPKVMLALY7DDB11A.jpg",
      "../dataset/table_rec_dataset_examples/images/no_border_5223_HLG406UK35UD5EUYC2AV.jpg"
    ],
    "val_samples": 100,
    "val_sample_paths": [
      "../dataset/table_rec_dataset_examples/images/border_2945_L7MSRHBZRW6Y347G39O6.jpg",
      "../dataset/table_rec_dataset_examples/images/no_border_288_6LK683JUCMOQ38V5BV29.jpg"
    ]
  },
  "analysis": {},
  "dataset_path": "./dataset/table_rec_dataset_examples",
  "show_type": "image",
  "dataset_type": "PubTabTableRecDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.train_samples：该数据集训练集样本数量为 2000；
- attributes.val_samples：该数据集验证集样本数量为 100；
- attributes.train_sample_paths：该数据集训练集样本可视化图片相对路径列表；
- attributes.val_sample_paths：该数据集验证集样本可视化图片相对路径列表；

**注**：只有通过数据校验的数据才可以训练和评估。


### 7.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，表格识别不支持格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，则需设置源数据集格式，表格识别不支持格式转换，默认为 `null`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 8. 时序预测任务模块数据校验

### 8.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了时序预测 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ts_dataset_examples.tar -C ./dataset/
```

### 8.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 12194,
    "train_table": [
      [
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      [
        "2016-07-01 00:00:00",
        5.827000141143799,
        2.009000062942505,
        1.5989999771118164,
        0.4620000123977661,
        4.203000068664552,
        1.3400000333786009,
        30.5310001373291
      ],
      [
        "2016-07-01 01:00:00",
        5.692999839782715,
        2.075999975204468,
        1.4919999837875366,
        0.4259999990463257,
        4.142000198364259,
        1.371000051498413,
        27.78700065612793
      ]
    ],
    "val_samples": 3484,
    "val_table": [
      [
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      [
        "2017-11-21 02:00:00",
        12.994000434875488,
        4.889999866485597,
        10.055999755859377,
        2.878000020980835,
        2.559000015258789,
        1.2489999532699585,
        4.7129998207092285
      ],
      [
        "2017-11-21 03:00:00",
        11.92199993133545,
        4.554999828338623,
        9.097000122070312,
        3.0920000076293945,
        2.559000015258789,
        1.2790000438690186,
        4.8540000915527335
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": ".\/dataset\/ts_dataset_examples",
  "show_type": "csv",
  "dataset_type": "TSDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.train_samples：该数据集训练集样本数量为 12194；
- attributes.val_samples：该数据集验证集样本数量为 3484；
- attributes.train_table：该数据集训练集样本示例数据表格信息；
- attributes.val_table：该数据集验证集样本示例数据表格信息；

**注**：只有通过数据校验的数据才可以训练和评估。


### 8.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，时序预测仅支持将xlsx标注文件转换为xls，无需设置源数据集格式，默认为 `null`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 9. 时序异常检测任务模块数据校验

### 9.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了时序异常检测 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar -P ./dataset
tar -xf ./dataset/ts_anomaly_examples.tar -C ./dataset/
```

### 9.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/DLinear_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 22032,
    "train_table": [
      [
        "timestamp",
        "feature_0",
        "...",
        "feature_24",
        "label"
      ],
      [
        0.0,
        0.7326893750079723,
        "...",
        0.1382488479262673,
        0.0
      ]
    ],
    "val_samples": 198290,
    "val_table": [
      [
        "timestamp",
        "feature_0",
        "...",
        "feature_24",
        "label"
      ],
      [
        22032.0,
        0.8604795809835284,
        "...",
        0.1428571428571428,
        0.0
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": "./dataset/ts_anomaly_examples",
  "show_type": "csv",
  "dataset_type": "TSADDataset"
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.train_samples：该数据集训练集样本数量为 22032；
- attributes.val_samples：该数据集验证集样本数量为 198290；
- attributes.train_table：该数据集训练集样本示例数据表格信息；
- attributes.val_table：该数据集验证集样本示例数据表格信息；

**注**：只有通过数据校验的数据才可以训练和评估。


### 9.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，时序异常检测仅支持将xlsx标注文件转换为xls，无需设置源数据集格式，默认为 `null`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。

## 10. 时序分类任务模块数据校验

### 10.1 数据准备

您需要按照 PaddleX 支持的数据格式要求准备数据，关于数据标注，您可以参考[PaddleX 数据标注](./annotation/README.md)，关于数据格式介绍，您可以参考[PaddleX 数据格式介绍](./dataset_format.md)，此处我们准备了时序分类 Demo 数据供您使用。

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_classify_examples.tar -P ./dataset
tar -xf ./dataset/ts_classify_examples.tar -C ./dataset/
```

### 10.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/ts_classify_examples/DLinear_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```

执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 82620,
    "train_table": [
      [
        "Unnamed: 0",
        "group_id",
        "dim_0",
        ...,
        "dim_60",
        "label",
        "time"
      ],
      [
        0.0,
        0.0,
        0.000949,
        ...,
        0.12107,
        1.0,
        0.0
      ]
    ],
    "val_samples": 83025,
    "val_table": [
      [
        "Unnamed: 0",
        "group_id",
        "dim_0",
        ...,
        "dim_60",
        "label",
        "time"
      ],
      [
        0.0,
        0.0,
        0.004578,
        ...,
        0.15728,
        1.0,
        0.0
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
上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：

- attributes.train_samples：该数据集训练集样本数量为 82620；
- attributes.val_samples：该数据集验证集样本数量为 83025；
- attributes.train_table：该数据集训练集样本示例数据表格信息；
- attributes.val_table：该数据集验证集样本示例数据表格信息；


另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：
![样本分布直方图](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/open_source/tutorials/data/dataset_check/ts_classify_examples/histogram.png)

**注**：只有通过数据校验的数据才可以训练和评估。


### 10.3 数据集格式转换/数据集划分（非必选）

如需对数据集格式进行转换或是重新划分数据集，可通过修改配置文件或是追加超参数的方式进行设置。

数据集校验相关的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
    * `convert`:
        * `enable`: 是否进行数据集格式转换，为 `True` 时进行数据集格式转换，默认为 `False`;
        * `src_dataset_type`: 如果进行数据集格式转换，时序异常检测仅支持将xlsx标注文件转换为xls，无需设置源数据集格式，默认为 `null`；
    * `split`:
        * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
        * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 `val_percent` 值加和为100；
        * `val_percent`: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 `train_percent` 值加和为100；

数据转换和数据划分支持同时开启，对于数据划分原有标注文件会被在原路径下重命名为 `xxx.bak`，以上参数同样支持通过追加命令行参数的方式进行设置，例如如重新划分数据集并设置训练集与验证集比例：`-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`。
