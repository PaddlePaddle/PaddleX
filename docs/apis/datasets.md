# Dataset reading

## paddlex.datasets. ImageNet
> **Used for image classification models**
```
paddlex.datasets.ImageNet(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=8, parallel_method='process', shuffle=False)
```
Read a classification dataset in ImageNet format and process samples accordingly. For the introduction to the ImageNet dataset format, see the following document: [Dataset Format Description] (../data/format/classification.md)

Example: [Code file] (https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py)

> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
> > * **file_list** (str): Describes a file path to a dataset image file and category ID (Each line of path in the text is a relative path relative to `data_dir`).
> >* **label_list** (str): Describes a category information file path contained in the dataset.
> >* **transforms** ( paddlex.cls.transforms): Preprocessing/enhancement operator for each sample in the dataset. For the details, see [paddlex.cls.transforms](./transforms/cls_transforms.md)
> >* **num_workers** (int|str)：The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', `num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, `num_workers` is 8, otherwise `num_workers` is a half of the number of CPU cores.
> >* **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 8 by default.
> >* **parallel_method** (str): Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
> >* **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.



## paddlex.datasets. VOCDetection
> **Used for object detection models**
```
paddlex.datasets. VOCDetection(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=100, parallel_method='process', shuffle=False)
```

> Read a detection dataset in PascalVOC format and process samples accordingly. For the introduction to the PascalVOC dataset format, see the following document: [Dataset Format Description] (../data/format/detection.md)

> Example: [Code file](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/yolov3_darknet53.py)

> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
> > * **file_list** (str): Describes a file path to a dataset image file and the corresponding annotation file (Each line of path in the text is a relative path relative to `data_dir`).
> > * **label_list** (str): Describes a category information file path contained in the dataset.
> > * **transforms** (paddlex.det.transforms): Preprocessing/enhancement operator for each sample in the dataset. For the details, see [paddlex.det.transforms](./transforms/det_transforms.md)
> > * **num_workers** (int|str)：The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', `num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, `num_workers` is 8, otherwise `num_workers` is a half of the number of CPU cores.` ``
> > * **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 100 by default.
> > * **parallel_method** (str): Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
> > * **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.



## paddlex.datasets. CocoDetection
> **Used for instance segmentation/object detection models**
```
paddlex.datasets. CocoDetection(data_dir, ann_file, transforms=None, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
```

> Read a detection dataset in MSCOCO format and process samples accordingly. A dataset in this format can also be applied to the training of instance segmentation models. For the introduction to the MSCOCO dataset format, see the following document: [Dataset Format Description](../data/format/instance_segmentation.md)

> Example: [Code file](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py)

> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
> > * **ann_file** (str): Dataset annotation file as an independent file in json format.
> > * **transforms** (paddlex.det.transforms): Preprocessing/enhancement operator for each sample in the dataset. For the details, see [paddlex.det.transforms] (./transforms/det_transforms.md).
> > * **num_workers** (int|str)：The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', 'num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, 'num_workers' is 8, otherwise 'num_workers' is a half of the number of CPU cores.
> > * **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 100 by default.
> > * **parallel_method** (str): Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
> > * **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.

## paddlex.datasets. SegDataset
> **Used for semantic segmentation models**
```
paddlex.datasets. SegDataset(data_dir, file_list, label_list, transforms=None, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
```

> Read a semantic segmentation task dataset and process samples accordingly. For the introduction to the semantic segmentation task dataset format, see the following document: [Dataset Format Description](../data/format/segmentation.md)

> Example: [Code file](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/unet.py)

> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
* **file_list** (str): Describes a file path to a dataset image file and the corresponding annotation file (Each line of path in the text is a relative path relative to data_dir`).`
* **label_list** (str): Describes a category information file path contained in the dataset.
* **transforms** ( paddlex . seg.transforms[): Preprocessing/enhancement operator for each sample in the dataset. For the details, see paddlex.seg.transforms. ](./transforms/seg_transforms.md)
* **num_workers** (int|str)：The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, num_workers is 8, otherwise num_workers is a half of the number of CPU cores.` ``
* **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 100 by default.
* **parallel_method** (str): Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
* **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.



## paddlex.datasets. EasyDataCls
> **Used for image classification models**
```
paddlex.datasets. EasyDataCls(data_dir, file_list, label_list, transforms=None, num_workers='auto', buffer_size=8, parallel_method='process', shuffle=False)
```

> Read an annotation image classification dataset on the EasyData platform and process samples accordingly.

> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
> > * **file_list** (str): Describes a file path to a dataset image file and the corresponding annotation file (Each line of path in the text is a relative path relative to `data_dir`).
> > * **label_list** (str):Describes a category information file path contained in the dataset.
> > * **transforms** (paddlex.seg.transforms): Preprocessing/enhancement operator for each sample in the dataset. For the details, see [paddlex.cls.transforms](./transforms/cls_transforms.md). 
> > * **num_workers** (int|str)：The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', `num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, `num_workers` is 8, otherwise `num_workers` is a half of the number of CPU cores.
> > * **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 8 by default.
> > * **parallel_method** (str):Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
> > * **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.



## paddlex.datasets. EasyDataDet
>**Used for object detection/instance segmentation models**
```
paddlex.datasets.EasyDataCls(data_dir, file_list, label_list, transforms=None, num_workers='auto', buffer_size=8, parallel_method='process', shuffle=False)
```

> Read a dataset in EasyData object detection/instance segmentation format and process samples accordingly. A dataset in this format can also be applied to the training of instance segmentation models.


> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
> > * **file_list** (str): Describes a file path to a dataset image file and the corresponding annotation file (Each line of path in the text is a relative path relative to `data_dir`).
> > * **label_list** (str): Describes a category information file path contained in the dataset.
> > * **transforms** (paddlex.seg.transforms): Preprocessing/enhancement operator for each sample in the dataset. For the details, see [paddlex.det.transforms] (./transforms/det_transforms.md)
> > * **num_workers** (int|str)：The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', `num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, `num_workers` is 8, otherwise `num_workers` is a half of the number of CPU cores.
> > * **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 100 by default.
> > * **parallel_method** (str): Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
> > * **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.



## paddlex.datasets. EasyDataSeg
> **Used for semantic segmentation models**
```
paddlex.datasets.EasyDataDet(data_dir, file_list, label_list, transforms=None, num_workers=‘auto’, buffer_size=100, parallel_method='process', shuffle=False)
```

> Read an EasyData semantic segmentation task dataset and process samples accordingly.


> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
> > * **file_list** (str): Describes a file path to a dataset image file and the corresponding annotation file (Each line of path in the text is a relative path relative to `data_dir`).
> > * **label_list** (str): Describes a category information file path contained in the dataset.
> > * **transforms** (paddlex.det.transforms): Preprocessing/enhancement operator for each sample in the dataset. For the details, see [paddlex.seg.transforms] (./transforms/seg_transforms.md).
> > * **num_workers** (int|str)：The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', `num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, `num_workers` is 8, otherwise `num_workers` is a half of the number of CPU cores.
> > * **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 100 by default.
> > * **parallel_method** (str): Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
> > * **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.

## paddlex.datasets. ChangeDetDataset
> **Used for semantic segmentation models for change detection**
```
paddlex.datasets. ChangeDetDataset(data_dir, file_list, label_list, transforms=None, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
```

> Read a semantic segmentation dataset for change detection and process samples accordingly. For an introduction to the change detection dataset format, see the following document: [Dataset Format Description] (../data/format/change_det.md)

> Example: [Code file](https://github.com/PaddlePaddle/PaddleX/blob/develop/examples/change_detection/train.py)

> **Parameters**

> > * **data_dir** (str): Directory path where the dataset is located.
> > * **file_list** (str): Describes a file path to dataset image 1 and 2 files and the corresponding annotation file (Each line of path in the text is a relative path relative to `data_dir`).
> > * **label_list** (str): Describes a category information file path contained in the dataset.
> > * **transforms** (paddlex.seg.transforms): Preprocessing/enhancement operator for each sample in the dataset. For the details, see [paddlex.seg.transforms](./transforms/seg_transforms.md).
> > * **num_workers** (int|str): The number of threads or processes during the preprocessing of samples in the dataset. It is 'auto' by default. When it is set to 'auto', `num_workers` is set according to the actual number of CPU cores of the system. If half of the number of CPU cores is greater than 8, `num_workers` is 8, otherwise `num_workers` is a half of the number of CPU cores.
> > * **buffer_size** (int): Queue cache length during the preprocessing of samples in the dataset. The unit is the number of samples. It is 100 by default.
> > * **parallel_method** (str): Parallel processing method during the preprocessing of samples in the dataset. Two methods including 'thread' thread and 'process' process are supported. It is 'process' by default (Thread is mandatory in Windows and Mac and this parameter is invalid).
> > * **shuffle** (bool): Whether to disrupt the order of the samples in the dataset. It is false by default.


