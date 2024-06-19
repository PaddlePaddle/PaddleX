# PaddleX 数据格式说明

众所周知，数据准备是AI任务开发流程中的重要一步，但是不同 AI 任务对数据准备的要求不尽相同，而且单个 AI 任务具有多种数据集，经常给大家带来困惑。

因此，PaddleX 针对常见 AI 任务，给出通用简明的数据集规范说明，涵盖数据集名称、组织结构、标注格式。

**请您在下面找到特定 AI 任务，参考说明准备数据，进而可以通过 PaddleX 的数据校验，最后完成全流程任务开发。**

请注意：

- 如果已有数据集不符合 PaddleX 的规范说明，请大家进行相应转换。

## 1. 图像分类任务模块

PaddleX 针对图像分类任务定义的数据集，名称是 **ClsDataset**，组织结构和标注格式如下。 

```plain
dataset_dir    # 数据集根目录，目录名称可以改变
├── images     # 图像的保存目录，目录名称可以改变，但要注意与train.txt、val.txt的内容对应
├── label.txt  # 标注id和类别名称的对应关系，文件名称不可改变。每行给出类别id和类别名称，内容举例：45 wallflower
├── train.txt  # 训练集标注文件，文件名称不可改变。每行给出图像路径和图像类别id，使用空格分隔，内容举例：images/image_06765.jpg 0
└── val.txt    # 验证集标注文件，文件名称不可改变。每行给出图像路径和图像类别id，使用空格分隔，内容举例：images/image_06767.jpg 10
```

请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar) 和 [图像分类任务数据集说明](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/training/single_label_classification/dataset.md)。

如果您已有数据集且数据集格式为如下格式，但是没有标注文件，可以使用[脚本](https://paddleclas.bj.bcebos.com/tools/create_cls_trainval_lists.py)将已有的数据集生成标注文件。

```plain
dataset_dir          # 数据集根目录，目录名称可以改变      
├── images           # 图像的保存目录，目录名称可以改变
   ├── train         # 训练集目录，目录名称可以改变
      ├── class0     # 类名字，最好是有意义的名字，否则生成的类别映射文件label.txt无意义
         ├── xxx.jpg # 图片，此处支持层级嵌套
         ├── xxx.jpg # 图片，此处支持层级嵌套
         ...      
      ├── class1     # 类名字，最好是有意义的名字，否则生成的类别映射文件label.txt无意义
      ...
   ├── val           # 验证集目录，目录名称可以改变
```

如果您使用的是 PaddleX 2.x版本的图像分类数据集，在经过训练集/验证集/测试集切分后，手动将 train_list.txt、val_list.txt、test_list.txt修改为train.txt、val.txt、test.txt，并且按照规则修改 label.txt 即可。

原版label.txt

```python
classname1
classname2
classname3
...
```

修改后的label.txt

```python
0 classname1
1 classname2
2 classname3
...
```

## 2. 目标检测任务模块

PaddleX 针对目标检测任务定义的数据集，名称是 **COCODetDataset**，组织结构和标注格式如下。

```plain
dataset_dir                  # 数据集根目录，目录名称可以改变
├── annotations              # 标注文件的保存目录，目录名称不可改变
│   ├── instance_train.json  # 训练集标注文件，文件名称不可改变，采用COCO标注格式
│   └── instance_val.json    # 验证集标注文件，文件名称不可改变，采用COCO标注格式
└── images                   # 图像的保存目录，目录名称不可改变
```

标注文件采用 COCO 格式。请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar) 和 [目标检测数据准备](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/data/PrepareDetDataSet.md)。


当大家使用的是 PaddleX 2.x 版本时的目标检测数据集时，请参考[数据格式转换]()，将 VOC 格式数据集转换为 COCO 数据集。

## 3. 实例分割任务模块

PaddleX 针对实例分割任务定义的数据集，名称是 **COCOInstSegDataset**，组织结构和标注格式如下。

```plain
dataset_dir                  # 数据集根目录，目录名称可以改变
├── annotations              # 标注文件的保存目录，目录名称不可改变
│   ├── instance_train.json  # 训练集标注文件，文件名称不可改变，采用COCO标注格式
│   └── instance_val.json    # 验证集标注文件，文件名称不可改变，采用COCO标注格式
└── images                   # 图像的保存目录，目录名称不可改变
```

标注文件采用 COCO 格式。请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar) 和 [目标检测数据准备](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/data/PrepareDetDataSet.md)。


当大家使用的是 PaddleX 2.x 版本时的实例分割数据集时，请参考[数据格式转换]()，将 VOC 格式数据集转换为 COCO 数据集。

**注：格式标注要求**

- 实例分割数据要求采用 COCO 数据格式标注出数据集中每张图像各个目标区域的像素边界和类别，采用 [x1,y1,x2,y2,...,xn,yn] 表示物体的多边形边界（segmentation）。其中，(xn,yn) 多边形各个角点坐标。标注信息存放到 annotations 目录下的 json 文件中，训练集 instance_train.json 和验证集 instance_val.json 分开存放。

- 如果你有一批未标注数据，我们推荐使用 LabelMe 进行数据标注。对于使用 LabelMe 标注的数据集，产线支持进行数据格式转换，请选择对应的格式后，点击「开始校验」按钮。

- 为确保格式转换顺利完成，请严格遵循示例数据集的文件命名和组织方式： [LabelMe 示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/instance_seg_labelme_examples.tar)。


## 4. 语义分割任务模块

PaddleX 针对图像分割任务定义的数据集，名称是**SegDataset**，组织结构和标注格式如下。

```plain
dataset_dir         # 数据集根目录，目录名称可以改变
├── annotations     # 存放标注图像的目录，目录名称可以改变，注意与标识文件的内容相对应
├── images          # 存放原始图像的目录，目录名称可以改变，注意与标识文件的内容相对应
├── train.txt       # 训练集标注文件，文件名称不可改变。每行是原始图像路径和标注图像路径，使用空格分隔，内容举例：images/P0005.jpg annotations/P0005.png
└── val.txt         # 验证集标注文件，文件名称不可改变。每行是原始图像路径和标注图像路径，使用空格分隔，内容举例：images/N0139.jpg annotations/N0139.png
```

标注图像是单通道灰度图或者单通道伪彩色图，建议使用PNG格式保存。标注图像中每种像素值代表一个类别，类别必须从0开始依次递增，例如0、1、2、3表示4种类别。标注图像的像素存储是8bit，所以标注类别最多支持256类。

请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar) 和 [图像语义分割数据准备](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/data/marker/marker_cn.md)。

**注：格式标注要求**

语义分割数据要求标注出数据集中每张图像不同类别所在的不同区域，产线要求的 Seg 格式数据集使用单通道的标注图片，每一种 `像素值` 代表一种类别，像素标注类别需要从0开始递增，例如0，1，2，3表示有4种类别。建议标注图像使用PNG无损压缩格式的图片，支持的标注类别最多为256类。在 `train.txt` 与 `val.txt` 中每行给出图像路径和标注图像路径，使用空格分隔。对于自定义标注的数据集，只需要标注前景目标并设置标注类别即可，其他像素默认作为背景。如需要手动标注背景区域，类别必须设置为 `_background_`，否则格式转换数据集会出现错误。对于图片中的噪声部分或不参与模型训练的部分，可以使用 `__ignore__` 类，模型训练时会自动跳过对应部分。

如果你有一批未标注数据，我们推荐使用 <a href="https://github.com/labelmeai/labelme" target="_blank">LabelMe</a> 进行数据标注。为确保格式转换顺利完成，请严格遵循示例数据集的文件命名和组织方式：
[LabelMe 示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_dataset_to_convert.tar)


## 5. 文本检测任务模块

PaddleX针对文本检测任务定义的数据集，名称是**TextDetDataset**，组织结构和标注格式如下。

```plain
dataset_dir     # 数据集根目录，目录名称可以改变
├── images      # 存放图像的目录，目录名称可以改变，但要注意和train.txt val.txt的内容对应
├── train.txt   # 训练集标注文件，文件名称不可改变，内容举例：images/img_0.jpg \t [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
└── val.txt     # 验证集标注文件，文件名称不可改变，内容举例：images/img_61.jpg \t [{"transcription": "TEXT", "points": [[31, 10], [310, 140], [420, 220], [310, 170]]}, {...}]
```

标注文件的每行内容是一张图像的路径和一个组成元素是字典的列表，路径和列表必须使用制表符’\t‘进行分隔，不可使用空格进行分隔。

对于组成元素是字典的列表，字典中 points 表示文本框的四个顶点的坐标（x, y），从左上角的顶点开始顺时针排；字典中`transcription`表示该文本框的文字，若`transcription 的`内容为“###”时，表示该文本框无效，不参与训练。

如果您使用了[PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/PPOCRLabel/README_ch.md)标注数据，只需要在完成数据集划分后将文字检测（det）目录中的`det_gt_train.txt`改名为`train.txt`、`det_gt_test.txt`改名为`val.txt`即可。

请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar) 和 [文本检测数据准备](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/dataset/ocr_datasets.md)。

## 6. 文本识别任务模块

PaddleX针对文字识别任务定义的数据集，名称是**MSTextRecDataset**，组织结构和标注格式如下。

```plain
dataset_dir      # 数据集根目录，目录名称可以改变
├── images       # 存放图像的目录，目录名称可以改变，但要注意和train.txt val.txt的内容对应
├── train.txt    # 训练集标注文件，文件名称不可改变，内容举例：images/111085122871_0.JPG \t 百度
├── val.txt      # 验证集标注文件，文件名称不可改变，内容举例：images/111085122871_0.JPG \t 百度
└── dict.txt     # 字典文件，文件名称不可改变。字典文件将所有出现的字符映射为字典的索引，每行为一个单字，内容举例：百
```

标注文件的每行内容是图像路径和文本内容，两者必须使用制表符’\t‘进行分隔，不可使用空格进行分隔。

如果您使用了[PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/PPOCRLabel/README_ch.md)标注数据，只需要在完成数据集划分后将文字识别（rec）目录中的`rec_gt_train.txt`改名为`train.txt`、`rec_gt_test.txt`改名为`val.txt`即可。

字典文件dict.txt的每行是一个单字，如"a"、"度"、"3"等，如下所示：

```plain
a
度
3
```

推荐使用 PP-OCR [默认字典](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt)﻿（右击链接下载即可） 并重命名为 `dict.txt` ，也可使用脚本 [gen_dict.py](https://paddleocr.bj.bcebos.com/script/gen_dict.py) 根据训练/评估数据自动生成字典：

```python
# 将脚本下载至 {dataset_dir} 目录下
wget https://paddleocr.bj.bcebos.com/script/gen_dict.py
# 执行转化，默认训练集标注文件为"train.txt", 验证集标注文件为"val.txt", 生成的字典文件为"dict.txt"
python gen_dict.py
```

请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar) 和 [文本识别数据准备](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/recognition.md#12-自定义数据集)。


## 7. 表格识别任务模块

PaddleX 针对表格识别任务定义的数据集，名称是 **PubTabTableRecDataset**，组织结构和标注格式如下。

```plain
dataset_dir    # 数据集根目录，目录名称可以改变
├── images     # 图像的保存目录，目录名称可以改变，但要注意和train.txt val.txt的内容对应
├── train.txt  # 训练集标注文件，文件名称不可改变，内容举例：{"filename": "images/border.jpg", "html": {"structure": {"tokens": ["<tr>", "<td", " colspan=\"3\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>"]}, "cells": [{"tokens": ["、", "自", "我"], "bbox": [[[5, 2], [231, 2], [231, 35], [5, 35]]]}, {"tokens": ["9"], "bbox": [[[168, 68], [231, 68], [231, 98], [168, 98]]]}]}, "gt": "<html><body><table><tr><td colspan=\"3\">、自我</td></tr><tr><td>Aghas</td><td>失吴</td><td>月，</td></tr><tr><td>lonwyCau</td><td></td><td>9</td></tr></table></body></html>"}
└── val.txt    # 验证集标注文件，文件名称不可改变，内容举例：{"filename": "images/no_border.jpg", "html": {"structure": {"tokens": ["<tr>", "<td", " colspan=\"2\"", ">", "</td>", "<td", " rowspan=\"2\"", ">", "</td>", "<td", " rowspan=\"2\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>"]}, "cells": [{"tokens": ["a", "d", "e", "s"], "bbox": [[[0, 4], [284, 4], [284, 34], [0, 34]]]}, {"tokens": ["$", "7", "5", "1", "8", ".", "8", "3"], "bbox": [[[442, 67], [616, 67], [616, 100], [442, 100]]]}]}, "gt": "<html><body><table><tr><td colspan=\"2\">ades</td><td rowspan=\"2\">研究中心主任滕建</td><td rowspan=\"2\">品、家居用品位居商</td></tr><tr><td>naut</td><td>则是创办思</td></tr><tr><td>各方意见建议，确保</td><td>9.66</td><td>道开业，负责</td><td>$7518.83</td></tr></table></body></html>"}
```

标注文件采用 PubTabNet 数据集格式进行标注，每行内容都是一个字典。

请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar) 和 [表格识别数据准备](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/table_recognition.md#1-数据准备)。

## 8. 时序预测任务模块

PaddleX 针对长时序预测任务定义的数据集，名称是**TSDataset**，组织结构和标注格式如下。

```plain
dataset_dir         # 数据集根目录，目录名称可以改变     
├── train.csv       # 训练集标注文件，文件名称不可改变。表头是每列的列名称，每一行是某一个时间点采集的数据。
├── val.csv         # 验证集标注文件，文件名称不可改变。表头是每列的列名称，每一行是某一个时间点采集的数据。
└── test.csv        # 测试集标注文件（可选），文件名称不可改变。表头是每列的列名称，每一行是某一个时间点采集的数据。
```

请大家参考上述规范准备数据，此外可以参考： [示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_dataset_examples.tar)和[TS数据准备](https://paddlets.readthedocs.io/zh_CN/latest/source/get_started/get_started.html#built-in-tsdataset)。


## 9. 时序异常检测任务模块

PaddleX 针对时序异常检测任务定义的数据集，名称是**TSADDataset**，组织结构和标注格式如下。

```shell
dataset_dir     # 数据集根目录，目录名称可以改变
├── train.csv   # 训练集文件，文件名称不可改变
├── val.csv     # 验证集文件，文件名称不可改变
└── test.csv    # 测试集文件，文件名称不可改变
```

时序异常检测和多模型融合时序异常检测要求的数据集格式，支持 xls、xlsx 格式的数据集转换为 csv 格式。你可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)。

## 10. 时序分类任务模块

PaddleX 针对时序异常检测任务定义的数据集，名称是**TSCLSataset**，组织结构和标注格式如下。

```shell
dataset_dir     # 数据集根目录，目录名称可以改变
├── train.csv   # 训练集文件，文件名称不可改变，群组编号名称固定为"group_id"，标签变量名称固定为"label"
├── val.csv     # 验证集文件，文件名称不可改变，群组编号名称固定为"group_id"，标签变量名称固定为"label"
└── test.csv    # 测试集文件，文件名称不可改变，群组编号名称固定为"group_id"，标签变量(可不包含)名称固定为"label"
```

时序分类要求的数据集格式，支持 xls、xlsx 格式的数据集转换为 csv 格式。你可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_classify_examples.tar)。