# 目标检测

| 模型               | Top1精度 | 模型大小 | GPU预测速度 | Arm预测速度 | 备注 |
| :----------------  | :------- | :------- | :---------  | :---------  |     |
| YOLOv3-MobileNetV1   |  97.5%   |   22M    | 10ms        |   200ms     |    |
| YOLOv3-MobileNetV3  |   |    |       |     |     |
| YOLOv3-DarkNet53 |    |    |     |   |     |
| FasterRCNN-ResNet50-FPN |    |   |     |   |    |
| FasterRCNN-ResNet101-FPN        |   |    |    |   |     |
| FasterRCNN-HRNet-FPN     |   |    |    |   |     |

将对应模型的训练代码保存到本地后，即可直接训练，训练代码会自动下载训练数据开始训练，如保存为`faster_r50_fpn.py`，如下命令即可开始训练
```
python faster_r50_fpn.py
```

- 针对自己的机器环境和数据，调整训练参数？先了解下PaddleX中训练参数。[——>>传送门]()
- 没有机器资源？使用AIStudio免费的GPU资源在线训练模型。[——>>传送门]()
