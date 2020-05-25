# 实例分割

## MaskRCNN类

```python
paddlex.det.MaskRCNN(num_classes=81, backbone='ResNet50', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[32, 64, 128, 256, 512])

```

> 构建MaskRCNN检测器。**注意在MaskRCNN中，num_classes需要设置为类别数+背景类，如目标包括human、dog两种，则num_classes需设为3，多的一种为背景background类别**

> **参数**

> > - **num_classes** (int): 包含了背景类的类别数。默认为81。
> > - **backbone** (str): MaskRCNN的backbone网络，取值范围为['ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd']。默认为'ResNet50'。
> > - **with_fpn** (bool): 是否使用FPN结构。默认为True。
> > - **aspect_ratios** (list): 生成anchor高宽比的可选值。默认为[0.5, 1.0, 2.0]。
> > - **anchor_sizes** (list): 生成anchor大小的可选值。默认为[32, 64, 128, 256, 512]。

#### train 训练接口

```python
train(self, num_epochs, train_dataset, train_batch_size=1, eval_dataset=None, save_interval_epochs=1, log_interval_steps=20, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=1.0/800, warmup_steps=500, warmup_start_lr=1.0 / 2400, lr_decay_epochs=[8, 11], lr_decay_gamma=0.1, metric=None, use_vdl=False, early_stop=False, early_stop_patience=5, resume_checkpoint=None)
```

> MaskRCNN模型的训练接口，函数内置了`piecewise`学习率衰减策略和`momentum`优化器。

> **参数**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。目前检测仅支持单卡评估，训练数据batch大小与显卡数量之商为验证数据batch大小。默认为1。
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认值为'output'。
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为None。
> > - **optimizer** (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
> > - **learning_rate** (float): 默认优化器的初始学习率。默认为0.00125。
> > - **warmup_steps** (int):  默认优化器进行warmup过程的步数。默认为500。
> > - **warmup_start_lr** (int): 默认优化器warmup的起始学习率。默认为1.0/2400。
> > - **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[8, 11]。
> > - **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认值为None。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认值为False。
> > - **early_stop** (float): 是否使用提前终止训练策略。默认值为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认值为5。
> > - **resume_checkpoint** (str): 恢复训练时指定上次训练保存的模型路径。若为None，则不会恢复训练。默认值为None。

#### evaluate 评估接口

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```

> MaskRCNN模型的评估接口，模型评估后会返回在验证集上的指标box_mmap(metric指定为COCO时)和相应的seg_mmap。

> **参数**
>
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **batch_size** (int): 验证数据批大小。默认为1。当前只支持设置为1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'; 如为COCODetection，则`metric`为'COCO'。
> > - **return_details** (bool): 是否返回详细信息。默认值为False。
> >
> **返回值**
>
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当return_details为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'和'segm_mmap'或者’bbox_map‘和'segm_map'，分别表示预测框和分割区域平均准确率平均值在各个IoU阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含关键字：'bbox'，对应元素预测框结果列表，每个预测结果由图像id、预测框类别id、预测框坐标、预测框得分；'mask'，对应元素预测区域结果列表，每个预测结果由图像id、预测区域类别id、预测区域坐标、预测区域得分；’gt‘：真实标注框和标注区域相关信息。

#### predict 预测接口

```python
predict(self, img_file, transforms=None)
```

> MaskRCNN模型预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在FasterRCNN.test_transforms和FasterRCNN.eval_transforms中。如未在训练时定义eval_dataset，那在调用预测predict接口时，用户需要再重新定义test_transforms传入给predict接口。

> **参数**
>
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值**
>
> > - **list**: 预测结果列表，列表中每个元素均为一个dict，key'bbox', 'mask', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、Mask信息，类别、类别id、置信度。其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。Mask信息为原图大小的二值图，1表示像素点属于预测类别，0表示像素点是背景。
