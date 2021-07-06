# Semantic Segmentation

## paddlex.seg.DeepLabv3p

```python
paddlex.seg.DeepLabV3P(num_classes=2, backbone='ResNet50_vd', use_mixed_loss=False, output_stride=8, backbone_indices=(0, 3), aspp_ratios=(1, 12, 24, 36), aspp_out_channels=256, align_corners=False)
```

> 构建DeepLabV3P分割器。

> **参数**
> > - **num_classes** (int): 类别数，默认为2。
> > - **backbone** (str): DeepLabv3+的backbone网络，实现特征图的计算，取值范围为['ResNet50_vd', 'ResNet101_vd']，默认为'ResNet50_vd'。
> > - **use_mixed_loss** (bool or List[tuple]): 是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxLoss，权重分别为0.8和0.2。如果为False，则仅使用CrossEntropyLoss。也可以以列表的形式自定义混合损失函数，列表的每一个元素为(损失函数类型，权重)元组，损失函数类型取值范围为['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']。
> > - **output_stride** (int): backbone 输出特征图相对于输入的下采样倍数，一般取值为8或16。默认为8。
> > - **backbone_indices** (tuple): backbone网络输出层的索引值。默认为(0, 3)。
> > - **assp_ratios** (tuple): assp模块中空洞卷积的空洞大小。默认为(1, 12, 24, 36)。
> > - **assp_out_channels** (int): assp模块输出通道数。默认为256。
> > - **align_corners** (bool): 网络中对特征图进行插值时是否将四个角落像素的中心对齐。若特征图尺寸为偶数，建议设为True。若特征图尺寸为奇数，建议设为False。默认为False。

### train

```python
train(num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, optimizer=None, save_interval_epochs=1, log_interval_steps=2, save_dir='output', pretrain_weights='CITYSCAPES', learning_rate=0.01, lr_decay_power=0.9, early_stop=False, early_stop_patience=5, use_vdl=True)
```

> DeepLabV3P模型的训练接口，函数内置了`polynomial`学习率衰减策略和`momentum`优化器。

> **参数**
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.dataset): 训练数据集。
> > - **train_batch_size** (int): 训练数据batch大小，默认为2。目前语义分割仅支持每张卡上batch大小为1进行评估，`train_batch_size`参数不影响评估时的batch大小。
> > - **eval_dataset** (paddlex.dataset): 评估数据集。
> > - **optimizer** (paddle.optimizer.Optimizer): 优化器。当该参数为None时，使用默认的优化器：使用paddle.optimizer.Momentum优化方法，paddle.optimizer.lr.PolynomialDecay学习率衰减策略。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认'output'
> > - **pretrain_weights** (str): 若指定为'.pdparams'文件时，则从文件加载模型权重；若为字符串'CITYSCAPES'，则自动下载在CITYSCAPES图片数据上预训练的模型权重；若为字符串'PascalVOC'，则自动下载在PascalVOC图片数据上预训练的模型权重；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为'CITYSCAPES'。
> > - **learning_rate** (float): 默认优化器的初始学习率。默认为0.01。
> > - **lr_decay_power** (float): 默认优化器学习率衰减指数。默认为0.9。
> > - **early_stop** (bool): 是否使用提前终止训练策略。默认为False。
> > - **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认为5。
> > - **use_vdl** (bool): 是否使用VisualDL进行可视化。默认为True。

### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

> DeepLabV3P模型评估接口。

>  **参数**
> > - **eval_dataset** (paddlex.datasets): 评估数据集。
> > - **batch_size** (int): 评估时的batch大小。默认1。
> > - **return_details** (bool): 是否返回详细信息。默认False。

> **返回值**
> >
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当`return_details`为False时，返回metrics。metrics为dict，包含关键字：'miou'、'category_iou'、'oacc'、'category_acc'和'kappa'，分别表示平均IoU、各类别IoU、总体准确率、各类别准确率和kappa系数。eval_details为dict，包含关键字：'confusion_matrix'，表示评估的混淆矩阵。

### predict

```
predict(self, img_file, transforms=None):
```

> DeepLabV3P模型预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在`DeepLabv3p.test_transforms`和`DeepLabv3p.eval_transforms`中。如未在训练时定义eval_dataset，那在调用预测`predict`接口时，用户需要再重新定义test_transforms传入给`predict`接口。

> **参数**
> >
> > - **img_file** (List[np.ndarray or str], str or np.ndarray): 预测图像或包含多张预测图像的列表，预测图像可以是路径或numpy数组(HWC排列，BGR格式)。
> > - **transforms** (paddlex.seg.transforms): 数据预处理操作。

> **返回值**
> >
> > - **dict** ｜ **List[dict]**: 如果输入为单张图像，返回dict。包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)。如果输入为多张图像，返回由每张图像预测结果组成的列表。


### analyze_sensitivity

```python
analyze_sensitivity(self, dataset, batch_size=8, criterion='l1_norm', save_dir='output')
```

> DeepLabV3P模型敏感度分析接口。

> **参数**
>
> > - **dataset** (paddlex.dataset): 用于评估模型在不同剪裁比例下精度损失的评估数据集。
> > - **batch_size** (int): 评估模型精度损失时的batch大小。默认为8。
> > - **criterion** ({'l1_norm', 'fpgm'}): 进行Filter粒度剪裁时评估，评估Filter重要性的范数标准。如果为'l1_norm'，采用L1-Norm标准。如果为'fpgm'，采用 [Geometric Median](https://arxiv.org/abs/1811.00250) 标准。
> > - **save_dir** (str): 计算的得到的sensetives文件的存储路径。

### prune

```python
prune(self, pruned_flops, save_dir=None)
```

> DeepLabV3P模型剪裁接口。

> **参数**
> > - **pruned_flops** (float): 每秒浮点数运算次数（FLOPs）的剪裁比例。
> > - **save_dir** (None or str): 剪裁后模型保存路径。如果为None，剪裁完成后不会对模型进行保存。默认为None。

### quant_aware_train

```python
quant_aware_train(self, num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, optimizer=None, save_interval_epochs=1, log_interval_steps=2, save_dir='output', learning_rate=.0001, lr_decay_power=0.9, early_stop=False, early_stop_patience=5, use_vdl=True, quant_config=None)
```

> DeepLabV3P模型在线量化接口。

> **参数**
>
> > - **num_epochs** (int): 训练迭代轮数。
> > - **train_dataset** (paddlex.dataset): 训练数据集。
> > - **train_batch_size** (int): 训练数据batch大小。同时作为验证数据batch大小。默认为2。
> > - **eval_dataset** (paddlex.dataset): 评估数据集。
> > - **optimizer** (paddle.optimizer.Optimizer): 优化器。当该参数为None时，使用默认的优化器：使用paddle.optimizer.Momentum优化方法，paddle.optimizer.lr.PolynomialDecay学习率衰减策略。
> > - **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
> > - **log_interval_steps** (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
> > - **save_dir** (str): 模型保存路径。默认'output'
> > - **learning_rate** (float): 默认优化器的初始学习率。默认为0.0001。
> > - **lr_decay_power** (float): 默认优化器学习率衰减指数。默认为0.9。
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

## paddlex.seg.BiSeNetV2

```python
paddlex.seg.BiSeNetV2(num_classes=2, use_mixed_loss=False, align_corners=False)
```

> 构建BiSeNetV2分割器

> **参数**
>
> > - **num_classes** (int): 类别数，默认为2。
> > - **use_mixed_loss** (bool or List[tuple]): 是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxLoss，权重分别为0.8和0.2。如果为False，则仅使用CrossEntropyLoss。也可以以列表的形式自定义混合损失函数，列表的每一个元素为(损失函数类型，权重)元组，损失函数类型取值范围为['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']。
> > - **align_corners** (bool): 网络中对特征图进行插值时是否将四个角落像素的中心对齐。若特征图尺寸为偶数，建议设为True。若特征图尺寸为奇数，建议设为False。默认为False。

> - train 训练接口说明同 [DeepLabV3P模型train接口](#train)（`pretrain_weights`取值范围为['CITYSCAPES', None]）
> - evaluate 评估接口说明同 [DeepLabV3P模型evaluate接口](#evaluate)
> - predict 预测接口说明同 [DeepLabV3P模型predict接口](#predict)
> - analyze_sensitivity 敏感度分析接口说明同 [DeepLabV3P模型analyze_sensivity接口](#analyze_sensitivity)
> - prune 剪裁接口说明同 [DeepLabV3P模型prune接口](#prune)
> - quant_aware_train 在线量化接口说明同 [DeepLabV3P模型quant_aware_train接口](#quant_aware_train)

## paddlex.seg.UNet

```python
paddlex.seg.UNet(num_classes=2, use_mixed_loss=False, use_deconv=False, align_corners=False)
```

> 构建UNet分割器。

> **参数**
>
> > - **num_classes** (int): 类别数，默认为2。
> > - **use_mixed_loss** (bool or List[tuple]): 是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxLoss，权重分别为0.8和0.2。如果为False，则仅使用CrossEntropyLoss。也可以以列表的形式自定义混合损失函数，列表的每一个元素为(损失函数类型，权重)元组，损失函数类型取值范围为['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']。
> > - **use_deconv** (bool): 在上采样过程中是否使用反卷积操作。如果为True，使用反卷积进行上采样。如果为False，使用双线性插值方法进行上采样。默认为False。
> > - **align_corners** (bool): 网络中对特征图进行插值时是否将四个角落像素的中心对齐。若特征图尺寸为偶数，建议设为True。若特征图尺寸为奇数，建议设为False。默认为False。

> - train 训练接口说明同 [DeepLabV3P模型train接口](#train)（`pretrain_weights`取值范围为['CITYSCAPES', None]）
> - evaluate 评估接口说明同 [DeepLabV3P模型evaluate接口](#evaluate)
> - predict 预测接口说明同 [DeepLabV3P模型predict接口](#predict)
> - analyze_sensitivity 敏感度分析接口说明同 [DeepLabV3P模型analyze_sensivity接口](#analyze_sensitivity)
> - prune 剪裁接口说明同 [DeepLabV3P模型prune接口](#prune)
> - quant_aware_train 在线量化接口说明同 [DeepLabV3P模型quant_aware_train接口](#quant_aware_train)

## paddlex.seg.HRNet

```python
paddlex.seg.HRNet(num_classes=2, width=48, use_mixed_loss=False, align_corners=False)
```

> 构建HRNet分割器。

> **参数**
>
> > - **num_classes** (int): 类别数，默认为2。
> > - **width** (int|str): 高分辨率分支中特征层的通道数量。默认为48。可选择取值为[18, 48]。
> > - **use_mixed_loss** (bool or List[tuple]): 是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxLoss，权重分别为0.8和0.2。如果为False，则仅使用CrossEntropyLoss。也可以以列表的形式自定义混合损失函数，列表的每一个元素为(损失函数类型，权重)元组，损失函数类型取值范围为['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']。
> > - **align_corners** (bool): 网络中对特征图进行插值时是否将四个角落像素的中心对齐。若特征图尺寸为偶数，建议设为True。若特征图尺寸为奇数，建议设为False。默认为False。

> - train 训练接口说明同 [DeepLabV3P模型train接口](#train)（`pretrain_weights`取值范围为['CITYSCAPES', 'PascalVOC', None]）
> - evaluate 评估接口说明同 [DeepLabV3P模型evaluate接口](#evaluate)
> - predict 预测接口说明同 [DeepLabV3P模型predict接口](#predict)
> - analyze_sensitivity 敏感度分析接口说明同 [DeepLabV3P模型analyze_sensivity接口](#analyze_sensitivity)
> - prune 剪裁接口说明同 [DeepLabV3P模型prune接口](#prune)
> - quant_aware_train 在线量化接口说明同 [DeepLabV3P模型quant_aware_train接口](#quant_aware_train)

## paddlex.seg.FastSCNN

```python
paddlex.seg.FastSCNN(num_classes=2, use_mixed_loss=False, align_corners=False)
```

> 构建FastSCNN分割器。

> **参数**
>
> > - **num_classes** (int): 类别数，默认为2。
> > - **use_mixed_loss** (bool or List[tuple]): 是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxLoss，权重分别为0.8和0.2。如果为False，则仅使用CrossEntropyLoss。也可以以列表的形式自定义混合损失函数，列表的每一个元素为(损失函数类型，权重)元组，损失函数类型取值范围为['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']。
> > - **align_corners** (bool): 网络中对特征图进行插值时是否将四个角落像素的中心对齐。若特征图尺寸为偶数，建议设为True。若特征图尺寸为奇数，建议设为False。默认为False。

> - train 训练接口说明同 [DeepLabV3P模型train接口](#train)（`pretrain_weights`取值范围为['CITYSCAPES', None]）
> - evaluate 评估接口说明同 [DeepLabV3P模型evaluate接口](#evaluate)
> - predict 预测接口说明同 [DeepLabV3P模型predict接口](#predict)
> - analyze_sensitivity 敏感度分析接口说明同 [DeepLabV3P模型analyze_sensivity接口](#analyze_sensitivity)
> - prune 剪裁接口说明同 [DeepLabV3P模型prune接口](#prune)
> - quant_aware_train 在线量化接口说明同 [DeepLabV3P模型quant_aware_train接口](#quant_aware_train)
