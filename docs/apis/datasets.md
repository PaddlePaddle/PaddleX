# 数据集读取

## 目录

* [ImageNet](#1)
* [VOCDetection](#2)
* [CocoDetection](#3)
* [SegDataset](#4)

## <h2 id="1">paddlex.datasets.ImageNet</h2>
> **用于图像分类模型**  
```python
paddlex.datasets.ImageNet(data_dir, file_list, label_list, transforms=None, num_workers='auto', shuffle=False)
```
读取ImageNet格式的分类数据集，并对样本进行相应的处理。ImageNet数据集格式的介绍可查看文档:[数据集格式说明](../data/format/classification.md)  

示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/tutorials/train/image_classification/mobilenetv3_small.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和类别id的文件路径（文本内每行路径为相对`data_dir`的相对路径）。  
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。  

## <h2 id="1">paddlex.datasets.VOCDetection</h2>
> **用于目标检测模型**  
```python
paddlex.datasets.VOCDetection(data_dir, file_list, label_list, transforms=None, num_workers='auto', shuffle=False)
```

> 读取PascalVOC格式的检测数据集，并对样本进行相应的处理。PascalVOC数据集格式的介绍可查看文档:[数据集格式说明](../data/format/detection.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/tutorials/train/object_detection/yolov3_darknet53.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。

## <h2 id="1">paddlex.datasets.CocoDetection</h2>
> **用于实例分割/目标检测模型**  
```python
paddlex.datasets.CocoDetection(data_dir, ann_file, transforms=None, num_workers='auto', shuffle=False)
```

> 读取MSCOCO格式的检测数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。MSCOCO数据集格式的介绍可查看文档:[数据集格式说明](../data/format/instance_segmentation.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **ann_file** (str): 数据集的标注文件，为一个独立的json格式文件。
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/transforms.md)。
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。

## <h2 id="1">paddlex.datasets.SegDataset</h2>
> **用于语义分割模型**  
```python
paddlex.datasets.SegDataset(data_dir, file_list, label_list=None, transforms=None, num_workers='auto', shuffle=False)
```

> 读取语义分割任务数据集，并对样本进行相应的处理。语义分割任务数据集格式的介绍可查看文档:[数据集格式说明](../data/format/segmentation.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/tutorials/train/semantic_segmentation/unet.py)

> **参数**
>
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.transforms](./transforms/seg_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。
