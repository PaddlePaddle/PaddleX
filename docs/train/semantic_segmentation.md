# 语义分割

| 模型               | Top1精度 | 模型大小 | GPU预测速度 | Arm预测速度 | 备注 |
| :----------------  | :------- | :------- | :---------  | :---------  |     |
| DeepLabv3p-MobileNetV2   |  97.5%   |   22M    | 10ms        |   200ms     |    |
| DeepLabv3p-Xception65  |   |    |       |     |     |
| UNet |    |    |     |   |     |
| HRNet |   |    |     |   |     |
| FastSCNN |  |   |    |    |    |

将对应模型的训练代码保存到本地后，即可直接训练，训练代码会自动下载训练数据开始训练，如保存为`deeplab_mobilenetv2.py`，如下命令即可开始训练
```
python deeplab_mobilenetv2.py
```

- 针对自己的机器环境和数据，调整训练参数？先了解下PaddleX中训练参数。[——>>传送门]()
- 没有机器资源？使用AIStudio免费的GPU资源在线训练模型。[——>>传送门]()
