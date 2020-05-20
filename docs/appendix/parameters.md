# 训练参数调整

PaddleX所有训练接口中，内置的参数均为根据单GPU卡相应batch_size下的较优参数，用户在自己的数据上训练模型，涉及到参数调整时，如无太多参数调优经验，则可参考如下方式

## 1.Epoch数的调整
Epoch数是模型训练过程，迭代的轮数，用户可以设置较大的数值，根据模型迭代过程在验证集上的指标表现，来判断模型是否收敛，进而提前终止训练。此外也可以使用`train`接口中的`early_stop`策略，模型在训练过程会自动判断模型是否收敛自动中止。

## 2.Batch Size的调整
Batch Size指模型在训练过程中，一次性处理的样本数量, 如若使用多卡训练， batch_size会均分到各张卡上（因此需要让batch size整除卡数）。这个参数跟机器的显存/内存高度相关，`batch_size`越高，所消耗的显存/内存就越高。PaddleX在各个`train`接口中均配置了默认的batch size，如若用户调整batch size，则也注意需要对应调整其它参数，如下表所示展示YOLOv3在训练时的参数配置

|       参数       |     默认值    |      调整比例       |      示例     |
|:---------------- | :------------ | :------------------ | :------------ |
| train_batch_size |      8        |   调整为 8*alpha    |      16       |
| learning_rate    |    1.0/8000   |   调整为 alpha/8000 |    2.0/8000   |
| warmup_steps     |    1000       |   调整为 1000/alpha<br>(该参数也可以自行根据数据情况调整) |     500       |
| lr_decay_epochs  | [213, 240]    |   不变              |   [213, 240]  |


更多训练接口可以参考
- [分类模型-train](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/classification.html#train)
- [目标检测检测FasterRCNN-train](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/detection.html#id2)
- [目标检测YOLOv3-train](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/detection.html#train)
- [实例分割MaskRCNN-train](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/instance_segmentation.html#train)
- [语义分割DeepLabv3p-train](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/semantic_segmentation.html#train)
- [语义分割UNet](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/semantic_segmentation.html#id2)
