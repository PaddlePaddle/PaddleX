# 检测和实例分割数据集

## VOCDetection类

```
paddlex.datasets.VOCDetection(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=100, parallel_method='thread', shuffle=False)
```

> 仅用于**目标检测**。读取PascalVOC格式的检测数据集，并对样本进行相应的处理。PascalVOC数据集格式的介绍可查看文档:[数据集格式说明](../datasets.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/detection/yolov3_darknet53.py#L29)

> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.det.transforms](./transforms/det_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。  

> 【可选】支持在训练过程中加入无目标真值的背景图片来减少背景误检，定义VOCDetection类后调用其成员函数`append_backgrounds`添加背景图片即可：
> ```
> append_backgrounds(image_dir)
> ```
> > 示例：[代码](../../tuning_strategy.html#id2)

> > **参数**

> > > * **image_dir** (str): 背景图片所在的目录路径。 

## CocoDetection类

```
paddlex.datasets.CocoDetection(data_dir, ann_file, transforms=None, num_workers='auto', buffer_size=100, parallel_method='thread', shuffle=False)
```

> 用于**目标检测或实例分割**。读取MSCOCO格式的检测数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。MSCOCO数据集格式的介绍可查看文档:[数据集格式说明](../datasets.md)  

> 示例：[代码文件](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/detection/mask_rcnn_r50_fpn.py#L27)

> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **ann_file** (str): 数据集的标注文件，为一个独立的json格式文件。
> > * **transforms** (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.det.transforms](./transforms/det_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。  
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。  

> 【可选】支持在训练过程中加入无目标真值的背景图片来减少背景误检，定义CocoDetection类后调用其成员函数`append_backgrounds`添加背景图片即可：
> ```
> append_backgrounds(image_dir)
> ```
> > 示例：[代码](../../tuning_strategy.html#id2)

> > **参数**

> > > * **image_dir** (str): 背景图片所在的目录路径。 

## EasyDataDet类

```
paddlex.datasets.EasyDataDet(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=100, parallel_method='thread', shuffle=False)
```

> 用于**目标检测或实例分割**。读取EasyData目标检测格式数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。EasyData目标检测或实例分割任务数据集格式的介绍可查看文档:[数据集格式说明](../datasets.md)  


> **参数**

> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对`data_dir`的相对路径）。
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  
> > * **transforms** (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子，详见[paddlex.det.transforms](./transforms/det_transforms.md)。  
> > * **num_workers** (int|str)：数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
> > * **buffer_size** (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。  
> > * **parallel_method** (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。  
> > * **shuffle** (bool): 是否需要对数据集中样本打乱顺序。默认为False。


> 【可选】支持在训练过程中加入无目标真值的背景图片来减少背景误检，定义EasyDataDet类后调用其成员函数`append_backgrounds`添加背景图片即可：
> ```
> append_backgrounds(image_dir)
> ```
> > 示例：[代码](../../tuning_strategy.html#id2)

> > **参数**

> > > * **image_dir** (str): 背景图片所在的目录路径。 

