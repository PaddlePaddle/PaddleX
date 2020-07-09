# 训练参数调整

PaddleX所有训练接口中，内置的参数均为根据单GPU卡相应batch_size下的较优参数，用户在自己的数据上训练模型，涉及到参数调整时，如无太多参数调优经验，则可参考如下方式

## 1.Epoch数的调整
Epoch数是模型训练过程，迭代的轮数，用户可以设置较大的数值，根据模型迭代过程在验证集上的指标表现，来判断模型是否收敛，进而提前终止训练。此外也可以使用`train`接口中的`early_stop`策略，模型在训练过程会自动判断模型是否收敛自动中止。

## 2.batch_size和learning_rate

> - Batch Size指模型在训练过程中，一次性处理的样本数量
> - 如若使用多卡训练， batch_size会均分到各张卡上（因此需要让batch size整除卡数）
> - Batch Size跟机器的显存/内存高度相关，`batch_size`越高，所消耗的显存/内存就越高
> - PaddleX在各个`train`接口中均配置了默认的batch size(默认针对单GPU卡)，如若训练时提示GPU显存不足，则相应调低BatchSize，如若GPU显存高或使用多张GPU卡时，可相应调高BatchSize。
> - **如若用户调整batch size，则也注意需要对应调整其它参数，特别是train接口中默认的learning_rate值**。如在YOLOv3模型中，默认`train_batch_size`为8，`learning_rate`为0.000125，当用户将模型在2卡机器上训练时，可以将`train_batch_size`调整为16, 那么同时`learning_rate`也可以对应调整为0.000125 * 2 = 0.00025

## 3.warmup_steps和warmup_start_lr

在训练模型时，一般都会使用预训练模型，例如检测模型在训练时使用backbone在ImageNet数据集上的预训练权重。但由于在自行训练时，自己的数据与ImageNet数据集存在较大的差异，可能会一开始由于梯度过大使得训练出现问题，因此可以在刚开始训练时，让学习率以一个较小的值，慢慢增长到设定的学习率。因此`warmup_steps`和`warmup_start_lr`就是这个作用，模型开始训练时，学习率会从`warmup_start_lr`开始，在`warmup_steps`个batch数据迭代后线性增长到设定的学习率。

> 例如YOLOv3的train接口，默认`train_batch_size`为8，`learning_rate`为0.000125, `warmup_steps`为1000， `warmup_start_lr`为0.0；在此参数配置下表示，模型在启动训练后，在前1000个step(每个step表示一个batch的数据，也就是8个样本)内，学习率会从0.0开始线性增长到设定的0.000125。

## 4.lr_decay_epochs和lr_decay_gamma

`lr_decay_epochs`用于让学习率在模型训练后期逐步衰减，它一般是一个list，如[6, 8, 10]，表示学习率在第6个epoch时衰减一次，第8个epoch时再衰减一次，第10个epoch时再衰减一次。每次学习率衰减为之前的学习率*lr_decay_gamma。

> 例如YOLOv3的train接口，默认`num_epochs`为270,`learning_rate`为0.000125， `lr_decay_epochs`为[213, 240]，`lr_decay_gamma`为0.1;在此参数配置下表示，模型在启动训练后，在前213个epoch中，训练时使用的学习率为0.000125，在第213至240个epoch之间，训练使用的学习率为0.000125*0.1，在240个epoch之后，使用的学习率为0.000125*0.1*0.1

## 5.参数设定时的约束
根据上述几个参数，可以了解到学习率的变化分为WarmUp热身阶段和Decay衰减阶段，
> - Wamup热身阶段：随着训练迭代，学习率从较低的值逐渐线性增长至设定的值，以step为单位
> - Decay衰减阶段：随着训练迭代，学习率逐步衰减，如每次衰减为之前的0.1， 以epoch为单位
> step与epoch的关系：1个epoch由多个step组成，例如训练样本有800张图像，`train_batch_size`为8, 那么每个epoch都要完整用这800张图片训一次模型，而每个epoch总共包含800//8即100个step

在PaddleX中，约束warmup必须在Decay之前结束，因此各参数设置需要满足下面条件
```
warmup_steps <= lr_decay_epochs[0] * num_steps_each_epoch
```
其中`num_steps_each_epoch`计算方式如下,
```
num_steps_each_eposh = num_samples_in_train_dataset // train_batch_size
```

因此，如若你在启动训练时，被提示`warmup_steps should be less than...`时，即表示需要根据上述公式调整你的参数啦，可以调整`lr_decay_epochs`或者是`warmup_steps`。

## 6.如何使用多GPU卡进行训练
在`import paddlex`前配置环境变量，代码如下
```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 使用第1张GPU卡进行训练
# 注意paddle或paddlex都需要在设置环境变量后再import
import paddlex as pdx
```

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' # 不使用GPU，使用CPU进行训练
import paddlex as pdx
```

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3' # 使用第1、2、4张GPU卡进行训练
import paddlex as pdx
```


## 相关模型接口

- 图像分类模型 [train接口](../apis/models/classification.html#train)
- FasterRCNN [train接口](../apis/models/detection.html#id1)
- YOLOv3 [train接口](../apis/models/detection.html#train)
- MaskRCNN [train接口](../apis/models/instance_segmentation.html#train)
- DeepLabv3p [train接口](../apis/models/semantic_segmentation.html#train)
