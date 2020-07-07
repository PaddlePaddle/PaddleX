# Object Detection

## paddlex.det.YOLOv3

```python
paddlex.det.YOLOv3(num_classes=80, backbone='MobileNetV1', anchors=None, anchor_masks=None, ignore_threshold=0.7, nms_score_threshold=0.01, nms_topk=1000, nms_keep_topk=100, nms_iou_threshold=0.45, label_smooth=False, train_random_shapes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
```

> 构建YOLOv3检测器。**注意在YOLOv3，num_classes不需要包含背景类，如目标包括human、dog两种，则num_classes设为2即可，这里与FasterRCNN/MaskRCNN有差别**

> **参数**
>
> > - **num_classes** (int): 类别数。默认为80。
> > - **backbone** (str): YOLOv3的backbone网络，取值范围为['DarkNet53', 'ResNet34', 'MobileNetV1', 'MobileNetV3_large']。默认为'MobileNetV1'。
> > - **anchors** (list|tuple): anchor框的宽度和高度，为None时表示使用默认值
> >                  [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
>                   [59, 119], [116, 90], [156, 198], [373, 326]]。
> > - **anchor_masks** (list|tuple): 在计算YOLOv3损失时，使用anchor的mask索引，为None时表示使用默认值
> >                    [[6, 7, 8], [3, 4, 5], [0, 1, 2]]。
> > - **ignore_threshold** (float): 在计算YOLOv3损失时，IoU大于`ignore_threshold`的预测框的置信度被忽略。默认为0.7。
> > - **nms_score_threshold** (float): 检测框的置信度得分阈值，置信度得分低于阈值的框应该被忽略。默认为0.01。
> > - **nms_topk** (int): 进行NMS时，根据置信度保留的最大检测框数。默认为1000。
> > - **nms_keep_topk** (int): 进行NMS后，每个图像要保留的总检测框数。默认为100。
> > - **nms_iou_threshold** (float): 进行NMS时，用于剔除检测框IOU的阈值。默认为0.45。
> > - **label_smooth** (bool): 是否使用label smooth。默认值为False。
> > - **train_random_shapes** (list|tuple): 训练时从列表中随机选择图像大小。默认值为[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]。

### train

```python
train(self, num_epochs, train_dataset, train_batch_size=8, eval_dataset=None, save_interval_epochs=20, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=1.0/8000, warmup_steps=1000, warmup_start_lr=0.0, lr_decay_epochs=[213, 240], lr_decay_gamma=0.1, metric=None, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5, resume_checkpoint=None)
```

> YOLOv3模型的训练接口，函数内置了`piecewise`学习率衰减策略和`momentum`优化器。

> **参数**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。目前检测仅支持单卡评估，训练数据batch大小与显卡数量之商为验证数据batch大小。默认值为8。
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为20。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认值为'output'。
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为字符串'COCO'，则自动下载在COCO数据集上预训练的模型权重；若为None，则不使用预训练模型。默认为None。
> > - **optimizer** (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
> > - **learning_rate** (float): 默认优化器的学习率。默认为1.0/8000。
> > - **warmup_steps** (int):  默认优化器进行warmup过程的步数。默认为1000。
> > - **warmup_start_lr** (int): 默认优化器warmup的起始学习率。默认为0.0。
> > - **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[213, 240]。
> > - **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认值为None。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认值为False。
> > - **sensitivities_file** (str): 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串'DEFAULT'，则自动下载在PascalVOC数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
> > - **eval_metric_loss** (float): 可容忍的精度损失。默认为0.05。
> > - **early_stop** (bool): 是否使用提前终止训练策略。默认值为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认值为5。
> > - **resume_checkpoint** (str): 恢复训练时指定上次训练保存的模型路径。若为None，则不会恢复训练。默认值为None。

### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```

> YOLOv3模型的评估接口，模型评估后会返回在验证集上的指标`box_map`(metric指定为'VOC'时)或`box_mmap`(metric指定为`COCO`时)。

> **参数**
>
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **batch_size** (int): 验证数据批大小。默认为1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'；如为COCODetection，则`metric`为'COCO'默认为None， 如为EasyData类型数据集，同时也会使用'VOC'。
> > - **return_details** (bool): 是否返回详细信息。默认值为False。
> >
>  **返回值**
>
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当`return_details`为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'或者’bbox_map‘，分别表示平均准确率平均值在各个阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含关键字：'bbox'，对应元素预测结果列表，每个预测结果由图像id、预测框类别id、预测框坐标、预测框得分；’gt‘：真实标注框相关信息。

### predict

```python
predict(self, img_file, transforms=None)
```

> YOLOv3模型预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在`YOLOv3.test_transforms`和`YOLOv3.eval_transforms`中。如未在训练时定义eval_dataset，那在调用预测`predict`接口时，用户需要再重新定义`test_transforms`传入给`predict`接口

> **参数**
>
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值**
>
> > - **list**: 预测结果列表，列表中每个元素均为一个dict，key包括'bbox', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、类别、类别id、置信度，其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。


## paddlex.det.FasterRCNN

```python
paddlex.det.FasterRCNN(num_classes=81, backbone='ResNet50', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[32, 64, 128, 256, 512])

```

> 构建FasterRCNN检测器。 **注意在FasterRCNN中，num_classes需要设置为类别数+背景类，如目标包括human、dog两种，则num_classes需设为3，多的一种为背景background类别**

> **参数**

> > - **num_classes** (int): 包含了背景类的类别数。默认为81。
> > - **backbone** (str): FasterRCNN的backbone网络，取值范围为['ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd', 'HRNet_W18']。默认为'ResNet50'。
> > - **with_fpn** (bool): 是否使用FPN结构。默认为True。
> > - **aspect_ratios** (list): 生成anchor高宽比的可选值。默认为[0.5, 1.0, 2.0]。
> > - **anchor_sizes** (list): 生成anchor大小的可选值。默认为[32, 64, 128, 256, 512]。

### train

```python
train(self, num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, save_interval_epochs=1, log_interval_steps=2,save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=0.0025, warmup_steps=500, warmup_start_lr=1.0/1200, lr_decay_epochs=[8, 11], lr_decay_gamma=0.1, metric=None, use_vdl=False, early_stop=False, early_stop_patience=5, resume_checkpoint=None)
```

> FasterRCNN模型的训练接口，函数内置了`piecewise`学习率衰减策略和`momentum`优化器。

> **参数**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。目前检测仅支持单卡评估，训练数据batch大小与显卡数量之商为验证数据batch大小。默认为2。
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认值为'output'。
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为字符串'COCO'，则自动下载在COCO数据集上预训练的模型权重（注意：暂未提供ResNet18的COCO预训练模型）；为None，则不使用预训练模型。默认为None。
> > - **optimizer** (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
> > - **learning_rate** (float): 默认优化器的初始学习率。默认为0.0025。
> > - **warmup_steps** (int):  默认优化器进行warmup过程的步数。默认为500。
> > - **warmup_start_lr** (int): 默认优化器warmup的起始学习率。默认为1.0/1200。
> > - **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[8, 11]。
> > - **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认值为None。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认值为False。
> > - **early_stop** (float): 是否使用提前终止训练策略。默认值为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认值为5。
> > - **resume_checkpoint** (str): 恢复训练时指定上次训练保存的模型路径。若为None，则不会恢复训练。默认值为None。

### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```

> FasterRCNN模型的评估接口，模型评估后会返回在验证集上的指标box_map(metric指定为’VOC’时)或box_mmap(metric指定为COCO时)。

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
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当`return_details`为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'或者’bbox_map‘，分别表示平均准确率平均值在各个IoU阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含关键字：'bbox'，对应元素预测结果列表，每个预测结果由图像id、预测框类别id、预测框坐标、预测框得分；’gt‘：真实标注框相关信息。

### predict

```python
predict(self, img_file, transforms=None)
```

> FasterRCNN模型预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在`FasterRCNN.test_transforms`和`FasterRCNN.eval_transforms`中。如未在训练时定义eval_dataset，那在调用预测`predict`接口时，用户需要再重新定义test_transforms传入给`predict`接口。

> **参数**
>
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值**
>
> > - **list**: 预测结果列表，列表中每个元素均为一个dict，key包括'bbox', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、类别、类别id、置信度，其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。
