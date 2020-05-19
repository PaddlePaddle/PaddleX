# 数据集转换

当前PaddleX GUI支持ImageNet格式的图像分类数据集、VOC格式的目标检测数据集、COCO格式的实例分割数据集、Seg格式的语义分割的数据集，当使用LabelMe、EasyData、标注精灵这3个工具标注数据时，PaddleX提供了相应接口可将数据转换成与PaddleX GUI想适配的数据集，使用方式如下所示：

```python
import paddlex as pdx

# 该接口实现LabelMe数据集到VOC数据集的转换。
# image_dir为图像文件存放的路径。
# json_dir为与每张图像对应的json文件的存放路径。
# dataset_save_dir为转换后数据集存放路径。
pdx.tools.labelme2voc(image_dir='labelme_imgs',
                      json_dir='labelme_jsons',
                      dataset_save_dir='voc_dataset')
```

可替换labelme2voc实现不同数据集间的转换，目前提供的转换接口如下：  

| 接口      | 转换关系 |
| :-------- | :------- |
| labelme2voc  | LabelMe数据集转换为VOC数据集   |
| labelme2coco  | LabelMe数据集转换为COCO数据集   |
| labelme2seg  | LabelMe数据集转换为Seg数据集  |
| easydata2imagenet | EasyData数据集转换为ImageNet数据集  |
| easydata2voc | EasyData数据集转换为VOC数据集  |
| easydata2coco | EasyData数据集转换为COCO数据集  |
| easydata2seg | EasyData数据集转换为Seg数据集  |
| jingling2seg | 标注精灵数据集转换为Seg数据集  |