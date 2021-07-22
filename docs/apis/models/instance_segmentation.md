# Instance Segmentation

## 目录
* [paddlex.det.MaskRCNN](#1)
  * [train](#11)
  * [evaluate](#12)
  * [predict](#13)
  * [analyze_sensitivity](#14)
  * [prune](#15)
  * [quant_aware_train](#16)


## <h2 id="1">paddlex.det.MaskRCNN</h2>

```python
paddlex.det.MaskRCNN(num_classes=80, backbone='ResNet50_vd', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[[32], [64], [128], [256], [512]], keep_top_k=100, nms_threshold=0.5, score_threshold=0.05, fpn_num_channels=256, rpn_batch_size_per_im=256, rpn_fg_fraction=0.5, test_pre_nms_top_n=None, test_post_nms_top_n=1000)
```

> 构建MaskRCNN检测器。

> **参数**
>
> > - **num_classes** (int): 类别数。默认为80。
> > - **backbone** (str): MaskRCNN的backbone网络，取值范围为['ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101', 'ResNet101_vd']。默认为'ResNet50_vd'。
> > - **with_fpn** (bool): 是否使用FPN结构。默认为True。
> > - **aspect_ratios** (list): 生成anchor高宽比的可选值。默认为[0.5, 1.0, 2.0]。
> > - **anchor_sizes** (list): 生成anchor大小的可选值。默认为[[32], [64], [128], [256], [512]]。
> > - **keep_top_k** (int): RCNN部分在进行非极大值抑制计算后，每张图像保留最多保存`keep_top_k`个检测框。默认为100。
> > - **nms_threshold** (float): RCNN部分在进行非极大值抑制时，用于剔除检测框所需的IoU阈值。默认为0.5。
> > - **score_threshold** (float): RCNN部分在进行非极大值抑制前，用于过滤掉低置信度边界框所需的置信度阈值。默认为0.05。
> > - **fpn_num_channels** (int): FPN部分特征层的通道数量。默认为256。
> > - **rpn_batch_size_per_im** (int): 训练阶段，RPN部分每张图片的正负样本的数量总和。默认为256。
> > - **rpn_fg_fraction** (float): 训练阶段，RPN部分每张图片的正负样本数量总和中正样本的占比。默认为0.5。
> > - **test_pre_nms_top_n** (int)：预测阶段，RPN部分做非极大值抑制计算的候选框的数量。若设置为None, 有FPN结构的话，`test_pre_nms_top_n`会被设置成6000, 无FPN结构的话，`test_pre_nms_top_n`会被设置成1000。默认为None。
> > - **test_post_nms_top_n** (int): 预测阶段，RPN部分做完非极大值抑制后保留的候选框的数量。默认为1000。

### <h3 id="11">train</h3>

```python
train(self, num_epochs, train_dataset, train_batch_size=64, eval_dataset=None, optimizer=None, save_interval_epochs=1, log_interval_steps=10, save_dir='output', pretrain_weights='IMAGENET', learning_rate=.001, warmup_steps=0, warmup_start_lr=0.0, lr_decay_epochs=(216, 243), lr_decay_gamma=0.1, metric=None, use_ema=False, early_stop=False, early_stop_patience=5, use_vdl=True)
```

> MaskRCNN模型的训练接口，函数内置了`piecewise`学习率衰减策略和`momentum`优化器。

> **参数**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.dataset): 训练数据集。
> > - **train_batch_size** (int): 训练数据batch大小，默认为64。目前实例分割仅支持单卡batch大小为1进行评估，`train_batch_size`参数不影响评估时的batch大小。
> > - **eval_dataset** (paddlex.dataset or None): 评估数据集。当该参数为None时，训练过程中不会进行模型评估。默认为None。
> > - **optimizer** (paddle.optimizer.Optimizer): 优化器。当该参数为None时，使用默认优化器：paddle.optimizer.lr.PiecewiseDecay衰减策略，paddle.optimizer.Momentum优化方法。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为10。
> > - **save_dir** (str): 模型保存路径。默认为'output'。
> > - **pretrain_weights** (str or None): 若指定为'.pdparams'文件时，则从文件加载模型权重；若为字符串’IMAGENET’，则自动下载在ImageNet图片数据上预训练的模型权重（仅包含backbone网络）；若为字符串’COCO’，则自动下载在COCO数据集上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
> > - **learning_rate** (float): 默认优化器的学习率。默认为0.001。
> > - **warmup_steps** (int):  默认优化器进行warmup过程的步数。默认为0。
> > - **warmup_start_lr** (int): 默认优化器warmup的起始学习率。默认为0.0。
> > - **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[216, 243]。
> > - **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
> > - **metric** ({'COCO', 'VOC', None}): 训练过程中评估的方式。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'；如为COCODetection，则`metric`为'COCO'。
> > - **use_ema** (bool): 是否使用指数衰减计算参数的滑动平均值。默认为False。
> > - **early_stop** (bool): 是否使用提前终止训练策略。默认为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认为5。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认为True。
> > - **resume_checkpoint** (str): 恢复训练时指定上次训练保存的模型路径，例如`output/maskrcnn_r50fpn/best_model`。若为None，则不会恢复训练。默认值为None。

### <h3 id="12">evaluate</h3>

```python
evaluate(self, eval_dataset, batch_size=1, metric=None, return_details=False)
```

> MaskRCNN模型的评估接口，模型评估后会返回在验证集上的指标`box_map`(metric指定为'VOC'时)或`box_mmap`(metric指定为`COCO`时)。

> **参数**
>
> > - **eval_dataset** (paddlex.dataset): 评估数据集。
> > - **batch_size** (int): 验证数据批大小。默认为1。
> > - **metric** ({'COCO', 'VOC', None}): 训练过程中评估的方式。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'；如为COCODetection，则`metric`为'COCO'默认为None， 如为EasyData类型数据集，同时也会使用'VOC'。
> > - **return_details** (bool): 是否返回详细信息。默认为False。
> >
>  **返回值**
>
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当`return_details`为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'或者’bbox_map‘，分别表示平均准确率平均值在各个阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含bbox和gt两个关键字。其中关键字bbox的键值是一个列表，列表中每个元素代表一个预测结果，一个预测结果是一个由图像id，预测框类别id, 预测框坐标，预测框得分组成的列表。而关键字gt的键值是真实标注框的相关信息。

### <h3 id="13">predict</h3>

```python
predict(self, img_file, transforms=None)
```

> MaskRCNN模型预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在`MaskRCNN.test_transforms`和`MaskRCNN.eval_transforms`中。如未在训练时定义eval_dataset，那在调用预测`predict`接口时，用户需要再重新定义`test_transforms`传入给`predict`接口

> **参数**
>
> > - **img_file** (List[np.ndarray or str], str or np.ndarray): 预测图像或包含多张预测图像的列表，预测图像可以是路径或numpy数组(HWC排列，BGR格式)。
> > - **transforms** (paddlex.transforms): 数据预处理操作。
>
> **返回值**
>
> > - **list**: 预测结果列表。如果输入为单张图像，列表中每个元素均为一个dict，键值包括'bbox', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、类别、类别id、置信度，其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。如果输入为多张图像，如果输入为多张图像，返回由每张图像预测结果组成的列表。

### <h3 id="14">analyze_sensitivity</h3>

```python
analyze_sensitivity(self, dataset, batch_size=8, criterion='l1_norm', save_dir='output')
```

> MaskRCNN模型敏感度分析接口。

> **参数**
>
> > - **dataset** (paddlex.dataset): 用于评估模型在不同剪裁比例下精度损失的评估数据集。
> > - **batch_size** (int): 评估模型精度损失时的batch大小。默认为8。
> > - **criterion** ({'l1_norm', 'fpgm'}): 进行Filter粒度剪裁时评估，评估Filter重要性的范数标准。如果为'l1_norm'，采用L1-Norm标准。如果为'fpgm'，采用 [Geometric Median](https://arxiv.org/abs/1811.00250) 标准。
> > - **save_dir** (str): 计算的得到的sensetives文件的存储路径。

### <h3 id="15">prune</h3>

```python
prune(self, pruned_flops, save_dir=None)
```

> MaskRCNN模型剪裁接口。

> **参数**
> > - **pruned_flops** (float): 每秒浮点数运算次数（FLOPs）的剪裁比例。
> > - **save_dir** (None or str): 剪裁后模型保存路径。如果为None，剪裁完成后不会对模型进行保存。默认为None。

### <h3 id="16">quant_aware_train</h3>

```python
quant_aware_train(self, num_epochs, train_dataset, train_batch_size=64, eval_dataset=None, optimizer=None, save_interval_epochs=1, log_interval_steps=10, save_dir='output', learning_rate=.00001, warmup_steps=0, warmup_start_lr=0.0, lr_decay_epochs=(216, 243), lr_decay_gamma=0.1, metric=None, use_ema=False, early_stop=False, early_stop_patience=5, use_vdl=True, quant_config=None)
```

> MaskRCNN模型在线量化接口。

> **参数**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.dataset): 训练数据集。
> > - **train_batch_size** (int): 训练数据batch大小。同时作为验证数据batch大小。默认为64。
> > - **eval_dataset** (paddlex.dataset): 评估数据集。当该参数为None时，训练过程中不会进行模型评估。默认为None。
> > - **optimizer** (paddle.optimizer.Optimizer): 优化器。当该参数为None时，使用默认优化器：paddle.optimizer.lr.PiecewiseDecay衰减策略，paddle.optimizer.Momentum优化方法。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代步数）。默认为10。
> > - **save_dir** (str): 模型保存路径。默认为'output'。
> > - **learning_rate** (float): 默认优化器的初始学习率。默认为0.00001。
> > - **warmup_steps** (int): 默认优化器的warmup步数，学习率将在设定的步数内，从warmup_start_lr线性增长至设定的learning_rate，默认为0。
> > - **warmup_start_lr**(float): 默认优化器的warmup起始学习率，默认为0.0。
> > - **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[216， 243]。
> > - **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
> > - **metric** ({'COCO', 'VOC', None}): 训练过程中评估的方式。默认为None，根据用户传入的Dataset自动选择，如为VOCDetection，则`metric`为'VOC'；如为COCODetection，则`metric`为'COCO'。
> > - **use_ema** (bool): 是否使用指数衰减计算参数的滑动平均值。默认为False。
> > - **early_stop** (bool): 是否使用提前终止训练策略。默认为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认为5。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认为True。
> > - **quant_config** (None or dict): 量化器配置。如果为None，使用默认配置。如果为dict，可配置项如下：
> >  ```python
> >  {
> >     # weight预处理方法，默认为None，代表不进行预处理；当需要使用`PACT`方法时设置为`"PACT"`
> >      'weight_preprocess_type': None,
> >
> >     # activation预处理方法，默认为None，代表不进行预处理`
> >     'activation_preprocess_type': None,
> >
> >     # weight量化方法, 默认为'channel_wise_abs_max', 此外还支持'channel_wise_abs_max'
> >     'weight_quantize_type': 'channel_wise_abs_max',
> >
> >     # activation量化方法, 默认为'moving_average_abs_max', 此外还支持'abs_max'
> >     'activation_quantize_type': 'moving_average_abs_max',
> >
> >     # weight量化比特数, 默认为 8
> >     'weight_bits': 8,
> >
> >     # activation量化比特数, 默认为 8
> >     'activation_bits': 8,
> >
> >     # 'moving_average_abs_max'的滑动平均超参, 默认为0.9
> >     'moving_rate': 0.9,
> >
> >     # 需要量化的算子类型
> >     'quantizable_layer_type': ['Conv2D', 'Linear']
> >  }
> >  ```
