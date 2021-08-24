# 数据集读取

## 目录

* [ImageNet](#1)
* [VOCDetection](#2)
  * [cluster_yolo_anchor](#21)
* [CocoDetection](#3)
  * [cluster_yolo_anchor](#31)
* [SegDataset](#4)

## <h2 id="1">paddlex.datasets.ImageNet</h2>
> **用于图像分类模型**  
```python
paddlex.datasets.ImageNet(data_dir, file_list, label_list, transforms=None, num_workers='auto', shuffle=False)
```
读取ImageNet格式的分类数据集，并对样本进行相应的处理。ImageNet数据集格式的介绍可查看文档:[数据集格式说明](../data/format/classification.md)  

示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv3_small.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和类别id的文件路径（文本内每行路径为相对`data_dir`的相对路径）。  
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。  

## <h2 id="2">paddlex.datasets.VOCDetection</h2>
> **用于目标检测模型**  
```python
paddlex.datasets.VOCDetection(data_dir, file_list, label_list, transforms=None, num_workers='auto', shuffle=False)
```

> 读取PascalVOC格式的检测数据集，并对样本进行相应的处理。PascalVOC数据集格式的介绍可查看文档:[数据集格式说明](../data/format/detection.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/yolov3_darknet53.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。
> > * **allow_empty** (bool): 是否加载负样本。默认为False。

### <h3 id="21">cluster_yolo_anchor</h3>

```python
cluster_yolo_anchor(num_anchors, image_size, cache=True, cache_path=None, iters=300, gen_iters=1000, thresh=.25)
```

> 分析数据集中所有图像的标签，聚类生成YOLO系列检测模型指定格式的anchor，返回结果按照由小到大排列。

> **注解**
>
> 自定义YOLO系列模型的`anchor`需要同时指定`anchor_masks`参数。`anchor_masks`参数为一个二维的列表，其长度等于模型backbone获取到的特征图数量（对于PPYOLO的MobileNetV3和ResNet18_vd，特征图数量为2，其余情况为3）。列表中的每一个元素也为列表，代表对应特征图上所检测的anchor编号。
> 以PPYOLO网络的默认参数`anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]`，`anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]`为例，代表在第一个特征图上检测尺度为`[116, 90], [156, 198], [373, 326]`的目标，在第二个特征图上检测尺度为`[30, 61], [62, 45], [59, 119]`的目标，以此类推。

> **参数**
>
> > * **num_anchors** (int): 生成anchor的数量。PPYOLO，当backbone网络为MobileNetV3或ResNet18_vd时通常设置为6，其余情况通常设置为9。对于PPYOLOv2、PPYOLOTiny、YOLOv3，通常设置为9。
> > * **image_size** (List[int] or int)：训练时网络输入的尺寸。如果为list，长度须为2，分别代表高和宽；如果为int，代表输入尺寸高和宽相同。
> > * **cache** (bool): 是否使用缓存。聚类生成anchor需要遍历数据集统计所有真值框的尺寸以及所有图片的尺寸，较为耗时。如果为True，会将真值框尺寸信息以及图片尺寸信息保存至`cache_path`路径下，若路径下已存缓存文件，则加载该缓存。如果为False，则不会保存或加载。默认为True。
> > * **cache_path** (None or str)：真值框尺寸信息以及图片尺寸信息缓存路径。 如果为None，则使用数据集所在的路径`data_dir`。默认为None。
> > * **iters** (int)：K-Means聚类算法迭代次数。
> > * **gen_iters** (int)：基因演算法迭代次数。
> > * **thresh** (float)：anchor尺寸与真值框尺寸之间比例的阈值。

**代码示例**
```python
import paddlex as pdx
from paddlex import transforms as T

# 下载和解压昆虫检测数据集
dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(dataset, path='./')

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
train_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/train_list.txt',
    label_list='insect_det/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/val_list.txt',
    label_list='insect_det/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

# 在训练集上聚类生成9个anchor
anchors = train_dataset.cluster_yolo_anchor(num_anchors=9, image_size=608)
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/train/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.det.PPYOLO(num_classes=num_classes,
                       backbone='ResNet50_vd_dcn',
                       anchors=anchors,
                       anchor_masks=anchor_masks)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/detection.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/parameters.md
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    pretrain_weights='COCO',
    learning_rate=0.005 / 12,
    warmup_steps=500,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    lr_decay_epochs=[85, 135],
    save_dir='output/ppyolo_r50vd_dcn',
    use_vdl=True)
```

## <h2 id="3">paddlex.datasets.CocoDetection</h2>
> **用于实例分割/目标检测模型**  
```python
paddlex.datasets.CocoDetection(data_dir, ann_file, transforms=None, num_workers='auto', shuffle=False)
```

> 读取MSCOCO格式的检测数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。MSCOCO数据集格式的介绍可查看文档:[数据集格式说明](../data/format/instance_segmentation.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **ann_file** (str): 数据集的标注文件，为一个独立的json格式文件。
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/transforms.md)。
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。
> > * **allow_empty** (bool): 是否加载负样本。默认为False。

### <h3 id="31">cluster_yolo_anchor</h3>

```python
cluster_yolo_anchor(num_anchors, image_size, cache=True, cache_path=None, iters=300, gen_iters=1000, thresh=.25)
```

> 分析数据集中所有图像的标签，聚类生成YOLO系列检测模型指定格式的anchor，返回结果按照由小到大排列。

> **注解**
>
> 自定义YOLO系列模型的`anchor`需要同时指定`anchor_masks`参数。`anchor_masks`参数为一个二维的列表，其长度等于模型backbone获取到的特征图数量（对于PPYOLO的MobileNetV3和ResNet18_vd，特征图数量为2，其余情况为3）。列表中的每一个元素也为列表，代表对应特征图上所检测的anchor编号。
> 以PPYOLO网络的默认参数`anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]`，`anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]`为例，代表在第一个特征图上检测尺度为`[116, 90], [156, 198], [373, 326]`的目标，在第二个特征图上检测尺度为`[30, 61], [62, 45], [59, 119]`的目标，以此类推。

> **参数**
>
> > * **num_anchors** (int): 生成anchor的数量。PPYOLO，当backbone网络为MobileNetV3或ResNet18_vd时通常设置为6，其余情况通常设置为9。对于PPYOLOv2、PPYOLOTiny、YOLOv3，通常设置为9。
> > * **image_size** (List[int] or int)：训练时网络输入的尺寸。如果为list，长度须为2，分别代表高和宽；如果为int，代表输入尺寸高和宽相同。
> > * **cache** (bool): 是否使用缓存。聚类生成anchor需要遍历数据集统计所有真值框的尺寸以及所有图片的尺寸，较为耗时。如果为True，会将真值框尺寸信息以及图片尺寸信息保存至`cache_path`路径下，若路径下已存缓存文件，则加载该缓存。如果为False，则不会保存或加载。默认为True。
> > * **cache_path** (None or str)：真值框尺寸信息以及图片尺寸信息缓存路径。 如果为None，则使用数据集所在的路径`data_dir`。默认为None。
> > * **iters** (int)：K-Means聚类算法迭代次数。
> > * **gen_iters** (int)：基因演算法迭代次数。
> > * **thresh** (float)：anchor尺寸与真值框尺寸之间比例的阈值。

**代码示例**
```python
import paddlex as pdx
from paddlex import transforms as T

# 下载和解压昆虫检测数据集
dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(dataset, path='./')

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
train_dataset = pdx.datasets.CocoDetection(
    data_dir='xiaoduxiong_ins_det/JPEGImages',
    ann_file='xiaoduxiong_ins_det/train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='xiaoduxiong_ins_det/JPEGImages',
    ann_file='xiaoduxiong_ins_det/val.json',
    transforms=eval_transforms)

# 在训练集上聚类生成9个anchor
anchors = train_dataset.cluster_yolo_anchor(num_anchors=9, image_size=608)
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/train/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.det.PPYOLO(num_classes=num_classes,
                       backbone='ResNet50_vd_dcn',
                       anchors=anchors,
                       anchor_masks=anchor_masks)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/detection.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/parameters.md
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    pretrain_weights='COCO',
    learning_rate=0.005 / 12,
    warmup_steps=500,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    lr_decay_epochs=[85, 135],
    save_dir='output/ppyolo_r50vd_dcn',
    use_vdl=True)
```

## <h2 id="4">paddlex.datasets.SegDataset</h2>
> **用于语义分割模型**  
```python
paddlex.datasets.SegDataset(data_dir, file_list, label_list=None, transforms=None, num_workers='auto', shuffle=False)
```

> 读取语义分割任务数据集，并对样本进行相应的处理。语义分割任务数据集格式的介绍可查看文档:[数据集格式说明](../data/format/segmentation.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/unet.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/seg_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。
