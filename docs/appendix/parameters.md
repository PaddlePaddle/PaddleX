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

## 关于lr_decay_epoch, warmup_steps等参数的说明

在PaddleX或其它深度学习模型的训练过程中，经常见到lr_decay_epoch, warmup_steps, warmup_start_lr等参数设置，下面介绍一些这些参数的作用。  

首先这些参数都是用于控制模型训练过程中学习率的变化方式，例如我们在训练时将learning_rate设为0.1, 通常情况，在模型的训练过程中，学习率一直以0.1不变训练下去, 但为了调出更好的模型效果，我们往往不希望学习率一直保持不变。

### warmup_steps和warmup_start_lr

我们在训练模型时，一般都会使用预训练模型，例如检测模型在训练时使用backbone在ImageNet数据集上的预训练权重。但由于在自行训练时，自己的数据与ImageNet数据集存在较大的差异，可能会一开始由于梯度过大使得训练出现问题，因此可以在刚开始训练时，让学习率以一个较小的值，慢慢增长到设定的学习率。因此`warmup_steps`和`warmup_start_lr`就是这个作用，模型开始训练时，学习率会从`warmup_start_lr`开始，在`warmup_steps`内线性增长到设定的学习率。

### lr_decay_epochs和lr_decay_gamma

`lr_decay_epochs`用于让学习率在模型训练后期逐步衰减，它一般是一个list，如[6, 8, 10]，表示学习率在第6个epoch时衰减一次，第8个epoch时再衰减一次，第10个epoch时再衰减一次。每次学习率衰减为之前的学习率*lr_decay_gamma

### Notice

在PaddleX中，限制warmup需要在第一个学习率decay衰减前结束，因此要满足下面的公式
```
warmup_steps <= lr_decay_epochs[0] * num_steps_each_epoch
```
其中公式中`num_steps_each_epoch = num_samples_in_train_dataset // train_batch_size`。  

>  因此如若在训练时PaddleX提示`warmup_steps should be less than xxx`时，即可根据上述公式来调整你的`lr_decay_epochs`或者是`warmup_steps`使得两个参数满足上面的条件

> - 图像分类模型 [train接口文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/classification.html#train)
> - FasterRCNN [train接口文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/detection.html#fasterrcnn)
> - YOLOv3 [train接口文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/detection.html#yolov3)
> - MaskRCNN [train接口文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/instance_segmentation.html#maskrcnn)
> - DeepLab [train接口文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/semantic_segmentation.html#deeplabv3p)
> - UNet [train接口文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/semantic_segmentation.html#unet)
> - HRNet [train接口文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/models/semantic_segmentation.html#hrnet)
