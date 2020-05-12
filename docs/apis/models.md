# 模型-models

## 分类模型

### ResNet50类

```python
paddlex.cls.ResNet50(num_classes=1000)
```

构建ResNet50分类器，并实现其训练、评估和预测。  

#### **参数:**

> - **num_classes** (int): 类别数。默认为1000。  

#### 分类器训练函数接口

> ```python
> train(self, num_epochs, train_dataset, train_batch_size=64, eval_dataset=None, save_interval_epochs=1, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=0.025, lr_decay_epochs=[30, 60, 90], lr_decay_gamma=0.1, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5)
> ```
>
> **参数：**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。同时作为验证数据batch大小。默认值为64。
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代步数）。默认为2。
> > - **save_dir** (str): 模型保存路径。
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
> > - **optimizer** (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
> > - **learning_rate** (float): 默认优化器的初始学习率。默认为0.025。
> > - **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[30, 60, 90]。
> > - **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认值为False。
> > - **sensitivities_file** (str): 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串'DEFAULT'，则自动下载在ImageNet图片数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
> > - **eval_metric_loss** (float): 可容忍的精度损失。默认为0.05。
> > - **early_stop** (float): 是否使用提前终止训练策略。默认值为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认值为5。

#### 分类器评估函数接口

> ```python
> evaluate(self, eval_dataset, batch_size=1, epoch_id=None, return_details=False)
> ```
>
> **参数：**
>
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **batch_size** (int): 验证数据批大小。默认为1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **return_details** (bool): 是否返回详细信息，默认False。
>
> **返回值：**
>
> > - **dict**: 当return_details为False时，返回dict, 包含关键字：'acc1'、'acc5'，分别表示最大值的accuracy、前5个最大值的accuracy。
> > - **tuple** (metrics, eval_details): 当`return_details`为True时，增加返回dict，包含关键字：'true_labels'、'pred_scores'，分别代表真实类别id、每个类别的预测得分。

#### 分类器预测函数接口

> ```python
> predict(self, img_file, transforms=None, topk=5)
> ```
>
> **参数：**
>
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.cls.transforms): 数据预处理操作。
> > - **topk** (int): 预测时前k个最大值。

> **返回值：**
>
> > - **list**: 其中元素均为字典。字典的关键字为'category_id'、'category'、'score'，
> >       分别对应预测类别id、预测类别标签、预测得分。

### 其它分类器类

除`ResNet50`外，`paddlex.cls`下还提供了`ResNet18`、`ResNet34`、`ResNet101`、`ResNet50_vd`、`ResNet101_vd`、`ResNet50_vd_ssld`、`ResNet101_vd_ssld`、`DarkNet53`、`MobileNetV1`、`MobileNetV2`、`MobileNetV3_small`、`MobileNetV3_large`、`MobileNetV3_small_ssld`、`MobileNetV3_large_ssld`、`Xception41`、`Xception65`、`Xception71`、`ShuffleNetV2`,  使用方式（包括函数接口和参数）均与`ResNet50`一致，各模型效果可参考[模型库](../model_zoo.md)中列表。



## 检测模型

### YOLOv3类

```python
paddlex.det.YOLOv3(num_classes=80, backbone='MobileNetV1', anchors=None, anchor_masks=None, ignore_threshold=0.7, nms_score_threshold=0.01, nms_topk=1000, nms_keep_topk=100, nms_iou_threshold=0.45, label_smooth=False, train_random_shapes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
```

构建YOLOv3检测器，并实现其训练、评估和预测。  

**参数:**

> - **num_classes** (int): 类别数。默认为80。
> - **backbone** (str): YOLOv3的backbone网络，取值范围为['DarkNet53', 'ResNet34', 'MobileNetV1', 'MobileNetV3_large']。默认为'MobileNetV1'。
> - **anchors** (list|tuple): anchor框的宽度和高度，为None时表示使用默认值
>                  [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
>                   [59, 119], [116, 90], [156, 198], [373, 326]]。
> - **anchor_masks** (list|tuple): 在计算YOLOv3损失时，使用anchor的mask索引，为None时表示使用默认值
>                    [[6, 7, 8], [3, 4, 5], [0, 1, 2]]。
> - **ignore_threshold** (float): 在计算YOLOv3损失时，IoU大于`ignore_threshold`的预测框的置信度被忽略。默认为0.7。
> - **nms_score_threshold** (float): 检测框的置信度得分阈值，置信度得分低于阈值的框应该被忽略。默认为0.01。
> - **nms_topk** (int): 进行NMS时，根据置信度保留的最大检测框数。默认为1000。
> - **nms_keep_topk** (int): 进行NMS后，每个图像要保留的总检测框数。默认为100。
> - **nms_iou_threshold** (float): 进行NMS时，用于剔除检测框IOU的阈值。默认为0.45。
> - **label_smooth** (bool): 是否使用label smooth。默认值为False。
> - **train_random_shapes** (list|tuple): 训练时从列表中随机选择图像大小。默认值为[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]。

#### YOLOv3训练函数接口

> ```python
> train(self, num_epochs, train_dataset, train_batch_size=8, eval_dataset=None, save_interval_epochs=20, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=1.0/8000, warmup_steps=1000, warmup_start_lr=0.0, lr_decay_epochs=[213, 240], lr_decay_gamma=0.1, metric=None, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5)
> ```
>
> **参数：**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。目前检测仅支持单卡评估，训练数据batch大小与显卡数量之商为验证数据batch大小。默认值为8。
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为20。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认值为'output'。
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为None。
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
> > - **early_stop** (float): 是否使用提前终止训练策略。默认值为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认值为5。

#### YOLOv3评估函数接口

> ```python
> evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
> ```
>
> **参数：**
>
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **batch_size** (int): 验证数据批大小。默认为1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'；如为COCODetection，则`metric`为'COCO'默认为None。
> > - **return_details** (bool): 是否返回详细信息。默认值为False。
> >
>  **返回值：**
>
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当`return_details`为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'或者’bbox_map‘，分别表示平均准确率平均值在各个阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含关键字：'bbox'，对应元素预测结果列表，每个预测结果由图像id、预测框类别id、预测框坐标、预测框得分；’gt‘：真实标注框相关信息。

#### YOLOv3预测函数接口

> ```python
> predict(self, img_file, transforms=None)
> ```
>
> **参数：**
>
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值：**
>
> > - **list**: 预测结果列表，列表中每个元素均为一个dict，key'bbox', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、类别、类别id、置信度，其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。

### FasterRCNN类

```python
paddlex.det.FasterRCNN(num_classes=81, backbone='ResNet50', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[32, 64, 128, 256, 512])

```

构建FasterRCNN检测器，并实现其训练、评估和预测。  

**参数:**

> - **num_classes** (int): 包含了背景类的类别数。默认为81。
> - **backbone** (str): FasterRCNN的backbone网络，取值范围为['ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd']。默认为'ResNet50'。
> - **with_fpn** (bool): 是否使用FPN结构。默认为True。
> - **aspect_ratios** (list): 生成anchor高宽比的可选值。默认为[0.5, 1.0, 2.0]。
> - **anchor_sizes** (list): 生成anchor大小的可选值。默认为[32, 64, 128, 256, 512]。

#### FasterRCNN训练函数接口

> ```python
> train(self, num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, save_interval_epochs=1, log_interval_steps=2,save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=0.0025, warmup_steps=500, warmup_start_lr=1.0/1200, lr_decay_epochs=[8, 11], lr_decay_gamma=0.1, metric=None, use_vdl=False, early_stop=False, early_stop_patience=5)
>
> ```
>
> **参数：**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。目前检测仅支持单卡评估，训练数据batch大小与显卡数量之商为验证数据batch大小。默认为2。
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认值为'output'。
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为None。
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

#### FasterRCNN评估函数接口

> ```python
> evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
>
> ```
>
> **参数：**
>
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **batch_size** (int): 验证数据批大小。默认为1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'; 如为COCODetection，则`metric`为'COCO'。
> > - **return_details** (bool): 是否返回详细信息。默认值为False。
> >
> **返回值：**
>
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当`return_details`为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'或者’bbox_map‘，分别表示平均准确率平均值在各个IoU阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含关键字：'bbox'，对应元素预测结果列表，每个预测结果由图像id、预测框类别id、预测框坐标、预测框得分；’gt‘：真实标注框相关信息。

#### FasterRCNN预测函数接口

> ```python
> predict(self, img_file, transforms=None)
>
> ```
>
> **参数：**
>
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值：**
>
> > - **list**: 预测结果列表，列表中每个元素均为一个dict，key'bbox', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、类别、类别id、置信度，其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。

### MaskRCNN类

```python
paddlex.det.MaskRCNN(num_classes=81, backbone='ResNet50', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[32, 64, 128, 256, 512])

```

构建MaskRCNN检测器，并实现其训练、评估和预测。  

**参数:**

> - **num_classes** (int): 包含了背景类的类别数。默认为81。
> - **backbone** (str): MaskRCNN的backbone网络，取值范围为['ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd']。默认为'ResNet50'。
> - **with_fpn** (bool): 是否使用FPN结构。默认为True。
> - **aspect_ratios** (list): 生成anchor高宽比的可选值。默认为[0.5, 1.0, 2.0]。
> - **anchor_sizes** (list): 生成anchor大小的可选值。默认为[32, 64, 128, 256, 512]。

#### MaskRCNN训练函数接口

> ```python
> train(self, num_epochs, train_dataset, train_batch_size=1, eval_dataset=None, save_interval_epochs=1, log_interval_steps=20, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=1.0/800, warmup_steps=500, warmup_start_lr=1.0 / 2400, lr_decay_epochs=[8, 11], lr_decay_gamma=0.1, metric=None, use_vdl=False, early_stop=False, early_stop_patience=5)
>
> ```
>
> **参数：**
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

#### MaskRCNN评估函数接口

> ```python
> evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
>
> ```
>
> **参数：**
>
> > - **eval_dataset** (paddlex.datasets): 验证数据读取器。
> > - **batch_size** (int): 验证数据批大小。默认为1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **metric** (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'; 如为COCODetection，则`metric`为'COCO'。
> > - **return_details** (bool): 是否返回详细信息。默认值为False。
> >
> **返回值：**
>
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当return_details为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'和'segm_mmap'或者’bbox_map‘和'segm_map'，分别表示预测框和分割区域平均准确率平均值在各个IoU阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含关键字：'bbox'，对应元素预测框结果列表，每个预测结果由图像id、预测框类别id、预测框坐标、预测框得分；'mask'，对应元素预测区域结果列表，每个预测结果由图像id、预测区域类别id、预测区域坐标、预测区域得分；’gt‘：真实标注框和标注区域相关信息。

#### MaskRCNN预测函数接口

> ```python
> predict(self, img_file, transforms=None)
>
> ```
>
> **参数：**
>
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值：**
>
> > - **list**: 预测结果列表，列表中每个元素均为一个dict，key'bbox', 'mask', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、Mask信息，类别、类别id、置信度，其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。

## 分割模型

### DeepLabv3p类

```python
paddlex.seg.DeepLabv3p(num_classes=2, backbone='MobileNetV2_x1.0', output_stride=16, aspp_with_sep_conv=True, decoder_use_sep_conv=True, encoder_with_aspp=True, enable_decoder=True, use_bce_loss=False, use_dice_loss=False, class_weight=None, ignore_index=255)

```

构建DeepLabv3p分割器，并实现其训练、评估和预测。

**参数：**

> - **num_classes** (int): 类别数。
> - **backbone** (str): DeepLabv3+的backbone网络，实现特征图的计算，取值范围为['Xception65', 'Xception41', 'MobileNetV2_x0.25', 'MobileNetV2_x0.5', 'MobileNetV2_x1.0', 'MobileNetV2_x1.5', 'MobileNetV2_x2.0']，'MobileNetV2_x1.0'。
> - **output_stride** (int): backbone 输出特征图相对于输入的下采样倍数，一般取值为8或16。默认16。
> - **aspp_with_sep_conv** (bool):  decoder模块是否采用separable convolutions。默认True。
> - **decoder_use_sep_conv** (bool)： decoder模块是否采用separable convolutions。默认True。
> - **encoder_with_aspp** (bool): 是否在encoder阶段采用aspp模块。默认True。
> - **enable_decoder** (bool): 是否使用decoder模块。默认True。
> - **use_bce_loss** (bool): 是否使用bce loss作为网络的损失函数，只能用于两类分割。可与dice loss同时使用。默认False。
> - **use_dice_loss** (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用，当`use_bce_loss`和`use_dice_loss`都为False时，使用交叉熵损失函数。默认False。
> - **class_weight** (list/str): 交叉熵损失函数各类损失的权重。当`class_weight`为list的时候，长度应为`num_classes`。当`class_weight`为str时， weight.lower()应为'dynamic'，这时会根据每一轮各类像素的比重自行计算相应的权重，每一类的权重为：每类的比例 * num_classes。class_weight取默认值None是，各类的权重1，即平时使用的交叉熵损失函数。
> - **ignore_index** (int): label上忽略的值，label为`ignore_index`的像素不参与损失函数的计算。默认255。

#### DeepLabv3训练函数接口

> ```python
> train(self, num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, eval_batch_size=1, save_interval_epochs=1, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=0.01, lr_decay_power=0.9, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5):
>
> ```
>
> **参数：**
> >
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。同时作为验证数据batch大小。默认2。
> > - **eval_dataset** (paddlex.datasets): 评估数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认'output'
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认'IMAGENET'。
> > - **optimizer** (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认的优化器：使用fluid.optimizer.Momentum优化方法，polynomial的学习率衰减策略。
> > - **learning_rate** (float): 默认优化器的初始学习率。默认0.01。
> > - **lr_decay_power** (float): 默认优化器学习率衰减指数。默认0.9。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认False。
> > - **sensitivities_file** (str): 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串'DEFAULT'，则自动下载在ImageNet图片数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
> > - **eval_metric_loss** (float): 可容忍的精度损失。默认为0.05。
> > - **early_stop** (float): 是否使用提前终止训练策略。默认值为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认值为5。

#### DeepLabv3评估函数接口

> ```python
> evaluate(self, eval_dataset, batch_size=1, epoch_id=None, return_details=False):
> ```

>  **参数：**
> >
> > - **eval_dataset** (paddlex.datasets): 评估数据读取器。
> > - **batch_size** (int): 评估时的batch大小。默认1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **return_details** (bool): 是否返回详细信息。默认False。

> **返回值：**
> >
> > - **dict**: 当`return_details`为False时，返回dict。包含关键字：'miou'、'category_iou'、'macc'、
> >   'category_acc'和'kappa'，分别表示平均iou、各类别iou、平均准确率、各类别准确率和kappa系数。
> > - **tuple** (metrics, eval_details)：当`return_details`为True时，增加返回dict (eval_details)，
> >   包含关键字：'confusion_matrix'，表示评估的混淆矩阵。

#### DeepLabv3预测函数接口

> ```
> predict(self, im_file, transforms=None):
> ```

> **参数：**
> >
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.seg.transforms): 数据预处理操作。

> **返回值：**
> >
> > - **dict**: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)。

### UNet类

```python
paddlex.seg.UNet(num_classes=2, upsample_mode='bilinear', use_bce_loss=False, use_dice_loss=False, class_weight=None, ignore_index=255)
```

构建UNet分割器，并实现其训练、评估和预测。


**参数：**

> - **num_classes** (int): 类别数。
> - **upsample_mode** (str): UNet decode时采用的上采样方式，取值为'bilinear'时利用双线行差值进行上菜样，当输入其他选项时则利用反卷积进行上菜样，默认为'bilinear'。
> - **use_bce_loss** (bool): 是否使用bce loss作为网络的损失函数，只能用于两类分割。可与dice loss同时使用。默认False。
> - **use_dice_loss** (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。当use_bce_loss和use_dice_loss都为False时，使用交叉熵损失函数。默认False。
> - **class_weight** (list/str): 交叉熵损失函数各类损失的权重。当`class_weight`为list的时候，长度应为`num_classes`。当`class_weight`为str时， weight.lower()应为'dynamic'，这时会根据每一轮各类像素的比重自行计算相应的权重，每一类的权重为：每类的比例 * num_classes。class_weight取默认值None是，各类的权重1，即平时使用的交叉熵损失函数。
> - **ignore_index** (int): label上忽略的值，label为`ignore_index`的像素不参与损失函数的计算。默认255。

#### Unet训练函数接口

> ```python
> train(self, num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, eval_batch_size=1, save_interval_epochs=1, log_interval_steps=2, save_dir='output', pretrain_weights='COCO', optimizer=None, learning_rate=0.01, lr_decay_power=0.9, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5):
> ```
>
> **参数：**
> >
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.datasets): 训练数据读取器。
> > - **train_batch_size** (int): 训练数据batch大小。同时作为验证数据batch大小。默认2。
> > - **eval_dataset** (paddlex.datasets): 评估数据读取器。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认'output'
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在COCO图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认'COCO'。
> > - **optimizer** (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认的优化器：使用fluid.optimizer.Momentum优化方法，polynomial的学习率衰减策略。
> > - **learning_rate** (float): 默认优化器的初始学习率。默认0.01。
> > - **lr_decay_power** (float): 默认优化器学习率衰减指数。默认0.9。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认False。
> > - **sensitivities_file** (str): 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串'DEFAULT'，则自动下载在ImageNet图片数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
> > - **eval_metric_loss** (float): 可容忍的精度损失。默认为0.05。
> > - **early_stop** (float): 是否使用提前终止训练策略。默认值为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认值为5。

#### Unet评估函数接口

> ```
> evaluate(self, eval_dataset, batch_size=1, epoch_id=None, return_details=False):
> ```

> **参数：**
> >
> > - **eval_dataset** (paddlex.datasets): 评估数据读取器。
> > - **batch_size** (int): 评估时的batch大小。默认1。
> > - **epoch_id** (int): 当前评估模型所在的训练轮数。
> > - **return_details** (bool): 是否返回详细信息。默认False。

> **返回值：**
> >
> > - **dict**: 当return_details为False时，返回dict。包含关键字：'miou'、'category_iou'、'macc'、
> >   'category_acc'和'kappa'，分别表示平均iou、各类别iou、平均准确率、各类别准确率和kappa系数。
> > - **tuple** (metrics, eval_details)：当return_details为True时，增加返回dict (eval_details)，
> >   包含关键字：'confusion_matrix'，表示评估的混淆矩阵。

#### Unet预测函数接口

> ```
> predict(self, im_file, transforms=None):
> ```

> **参数：**
> >
> > - **img_file** (str): 预测图像路径。
> > - **transforms** (paddlex.seg.transforms): 数据预处理操作。

> **返回值：**
> >
> > - **dict**: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)。
