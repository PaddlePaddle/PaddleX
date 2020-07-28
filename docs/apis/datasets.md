# 数据集读取

## paddlex.datasets.ImageNet
> **用于图像分类模型**  
```
paddlex.datasets.ImageNet(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=8, parallel_method='process', shuffle=False)
```
读取ImageNet格式的分类数据集，并对样本进行相应的处理。ImageNet数据集格式的介绍可查看文档:[数据集格式说明](../data/format/index.html)  

示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py)

> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和类别id的文件路径（文本内每行路径为相对`data_dir`的相对路径）。  
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.cls.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.cls.transforms](./transforms/cls_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。  
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为8。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。  

## paddlex.datasets.VOCDetection
> **用于目标检测模型**  
```
paddlex.datasets.VOCDetection(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=100, parallel_method='process', shuffle=False)
```

> 读取PascalVOC格式的检测数据集，并对样本进行相应的处理。PascalVOC数据集格式的介绍可查看文档:[数据集格式说明](../data/format/index.html)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/yolov3_darknet53.py)

> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.det.transforms](./transforms/det_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。  

## paddlex.datasets.CocoDetection
> **用于实例分割/目标检测模型**  
```
paddlex.datasets.CocoDetection(data_dir, ann_file, transforms=None, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
```

> 读取MSCOCO格式的检测数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。MSCOCO数据集格式的介绍可查看文档:[数据集格式说明](../data/format/index.html)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py)

> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **ann_file** (str): 数据集的标注文件，为一个独立的json格式文件。
> > * **transforms** (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.det.transforms](./transforms/det_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。  
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。  

## paddlex.datasets.SegDataset
> **用于语义分割模型**  
```
paddlex.datasets.SegDataset(data_dir, file_list, label_list, transforms=None, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
```

> 读取语义分割任务数据集，并对样本进行相应的处理。语义分割任务数据集格式的介绍可查看文档:[数据集格式说明](../data/format/index.html)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/unet.py)

> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.seg.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.seg.transforms](./transforms/seg_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。

## paddlex.datasets.EasyDataCls
> **用于图像分类模型**  
```
paddlex.datasets.EasyDataCls(data_dir, file_list, label_list, transforms=None, num_workers='auto', buffer_size=8, parallel_method='process', shuffle=False)
```

> 读取EasyData平台标注图像分类数据集，并对样本进行相应的处理。

> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.seg.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.cls.transforms](./transforms/cls_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为8。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。

## paddlex.datasets.EasyDataDet
> 用于**目标检测/实例分割模型**  
```
paddlex.datasets.EasyDataDet(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=100, parallel_method='process', shuffle=False)
```

> 读取EasyData目标检测/实例分割格式数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。


> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.det.transforms](./transforms/det_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。

## paddlex.datasets.EasyDataSeg
> **用于语义分割模型**  
```
paddlex.datasets.EasyDataSeg(data_dir, file_list, label_list, transforms=None, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
```

> 读取EasyData语义分割任务数据集，并对样本进行相应的处理。


> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.seg.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.seg.transforms](./transforms/seg_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。
