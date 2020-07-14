# 图像分类

## 介绍

PaddleX共提供了20+的图像分类模型，可满足开发者不同场景的需求下的使用。

- **Top1精度**: 模型在ImageNet数据集上的测试精度
- **预测速度**：单张图片的预测用时（不包括预处理和后处理)
- "-"表示指标暂未更新

| 模型（点击获取代码）               | Top1精度 | 模型大小 | GPU预测速度 | Arm预测速度 | 备注 |
| :----------------  | :------- | :------- | :---------  | :---------  | :-----    |
| [MobileNetV3_small_ssld](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv3_small_ssld.py) |  71.3%  |  21.0MB  |  6.809ms   | -  |  模型小，预测速度快，适用于低性能或移动端设备   |
| [MobileNetV2](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py)        | 72.2%  | 14.0MB   |  4.546ms  | -  |  模型小，预测速度快，适用于低性能或移动端设备   |
| [ShuffleNetV2](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/shufflenetv2.py)     | 68.8%  | 9.0MB   | 6.101ms   | -  |  模型体积小，预测速度快，适用于低性能或移动端设备   |
| [ResNet50_vd_ssld](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/resnet50_vd_ssld.py)   |  82.4%   |   102.8MB    |  9.058ms       |   -    | 模型精度高，适用于服务端部署   |


## 开始训练

将代码保存到本地后运行（代码下载链接位于上面的表格），**代码会自动下载训练数据并开始训练**。如保存为`mobilenetv3_small_ssld.py`，执行如下命令即可开始训练：

```
python mobilenetv3_small_ssld.py
```


## 相关文档

- 【**重要**】针对自己的机器环境和数据，调整训练参数？先了解下PaddleX中训练参数作用。[——>>传送门](../appendix/parameters.md)
- 【**有用**】没有机器资源？使用AIStudio免费的GPU资源在线训练模型。[——>>传送门](https://aistudio.baidu.com/aistudio/projectdetail/450925)
- 【**拓展**】更多图像分类模型，查阅[PaddleX模型库](../appendix/model_zoo.md)和[API使用文档](../apis/models/index.html)。
