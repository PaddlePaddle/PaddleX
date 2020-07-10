# 实例分割

## 介绍

PaddleX目前提供了MaskRCNN实例分割模型结构,多种backbone模型，可满足开发者不同场景和性能的需求。

- **Box MMAP/Seg MMAP**: 模型在COCO数据集上的测试精度
- **预测速度**：单张图片的预测用时（不包括预处理和后处理)
- "-"表示指标暂未更新

| 模型(点击获取代码)               | Box MMAP/Seg MMAP | 模型大小 | GPU预测速度 | Arm预测速度 | 备注 |
| :----------------  | :------- | :------- | :---------  | :---------  | :-----    |
| [MaskRCNN-ResNet50-FPN](https://github.com/PaddlePaddle/PaddleX/blob/doc/tutorials/train/instance_segmentation/mask_r50_fpn.py)   |  -/-   |   136.0MB    |  197.715ms       |   -    | 模型精度高，适用于服务端部署   |
| [MaskRCNN-ResNet18-FPN](https://github.com/PaddlePaddle/PaddleX/blob/doc/tutorials/train/instance_segmentation/mask_r18_fpn.py)   |  -/-   |   -    |  -       |   -    | 模型精度高，适用于服务端部署   |
| [MaskRCNN-HRNet-FPN](https://github.com/PaddlePaddle/PaddleX/blob/doc/tutorials/train/instance_segmentation/mask_hrnet_fpn.py)   |  -/-   |   115.MB    |  81.592ms       |   -    | 模型精度高，预测速度快，适用于服务端部署   |


## 开始训练

> 代码保存到本地后，即可直接训练，**训练代码会自动下载训练数据开始训练**
> > 如保存为`mask_r50_fpn.py`，如下命令即可开始训练
> > ```
> > python mask_r50_fpn.py
> > ```

## 相关文档

- 【**重要**】针对自己的机器环境和数据，调整训练参数？先了解下PaddleX中训练参数作用。[——>>传送门](../appendix/parameters.md)
- 【**有用**】没有机器资源？使用AIStudio免费的GPU资源在线训练模型。[——>>传送门](https://aistudio.baidu.com/aistudio/projectdetail/450925)
- 【**拓展**】更多图像分类模型，查阅[PaddleX模型库](../appendix/model_zoo.md)和[API使用文档](../apis/models/index.html)。
