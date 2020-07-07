# 图像分类

PaddleX共提供了20+的图像分类模型，包括基于大规模数据训练的

点击表格中模型名，可获取各模型训练的教程代码

| 模型               | Top1精度 | 模型大小 | GPU预测速度 | Arm预测速度 | 备注 |
| :----------------  | :------- | :------- | :---------  | :---------  |     |
| ResNet50_vd_ssld   |  97.5%   |   22M    | 10ms        |   200ms     |    |
| ResNet101_vd_ssld  |   |    |       |     |     |
| MobileNetV3_small_ssld |    |    |     |   |     |
| MobileNetV3_large_ssld |    |   |     |   |    |
| MobileNetV2        |   |    |    |   |     |
| ShuffleNetV2     |   |    |    |   |     |
| AlexNet |    |      |      |    |     |


将对应模型的训练代码保存到本地后，即可直接训练，训练代码会自动下载训练数据开始训练，如保存为`resnet50_vd_ssld.py`，如下命令即可开始训练
```
python resnet50_vd_ssld.py
```

- 针对自己的机器环境和数据，调整训练参数？先了解下PaddleX中训练参数。[——>>传送门]()
- 没有机器资源？使用AIStudio免费的GPU资源在线训练模型。[——>>传送门]()
