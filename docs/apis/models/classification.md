# Image Classification

## 目录
* [paddlex.cls.ResNet50](#1)
  * [train](#11)
  * [evaluate](#12)
  * [predict](#13)
  * [analyze_sensitivity](#14)
  * [prune](#15)
  * [quant_aware_train](#16)
* [其他分类模型](#2)

## <h2 id="1">paddlex.cls.ResNet50</h2>

```python
paddlex.cls.ResNet50(num_classes=1000)
```

> 构建ResNet50分类器，并实现其训练、评估和预测。

> **参数**
>
> - **num_classes** (int): 类别数。默认为1000。  

### <h3 id="11">train</h3>

```python
train(self, num_epochs, train_dataset, train_batch_size=64, eval_dataset=None, optimizer=None, save_interval_epochs=1, log_interval_steps=10, save_dir='output', pretrain_weights='IMAGENET', learning_rate=.025, warmup_steps=0, warmup_start_lr=0.0, lr_decay_epochs=(30, 60, 90), lr_decay_gamma=0.1, early_stop=False, early_stop_patience=5, use_vdl=True)
```
>
> **参数**
>
- **num_epochs** (int): 训练迭代轮数。
- **train_dataset** (paddlex.dataset): 训练数据集。
- **train_batch_size** (int): 训练数据batch大小。同时作为验证数据batch大小。默认为64。
- **eval_dataset** (paddlex.dataset or None): 评估数据集。当该参数为None时，训练过程中不会进行模型评估。默认为None。
- **optimizer** (paddle.optimizer.Optimizer): 优化器。当该参数为None时，使用默认优化器：paddle.optimizer.lr.PiecewiseDecay衰减策略，paddle.optimizer.Momentum优化方法。
- **save_interval_epochs** (int): 模型保存间隔（单位：迭代轮数）。默认为1。
- **log_interval_steps** (int): 训练日志输出间隔（单位：迭代步数）。默认为10。
- **save_dir** (str): 模型保存路径。默认为'output'。
- **pretrain_weights** (str or None): 若指定为'.pdparams'文件时，则从文件加载模型权重；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
- **learning_rate** (float): 默认优化器的初始学习率。默认为0.025。
- **warmup_steps** (int): 默认优化器的warmup步数，学习率将在设定的步数内，从warmup_start_lr线性增长至设定的learning_rate，默认为0。
- **warmup_start_lr**(float): 默认优化器的warmup起始学习率，默认为0.0。
- **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[30, 60, 90]。
- **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
- **early_stop** (bool): 是否使用提前终止训练策略。默认为False。
- **early_stop_patience** (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内连续下降或持平，则终止训练。默认为5。
- **use_vdl** (bool): 是否使用VisualDL进行可视化。默认为True。
- **resume_checkpoint** (str): 恢复训练时指定上次训练保存的模型路径，例如`output/mobilenetv3_small/best_model`。若为None，则不会恢复训练。默认值为None。**恢复训练需要将`pretrain_weights`设置为None。**

### <h3 id="12">evaluate</h3>

```python
evaluate(self, eval_dataset, batch_size=1, return_details=False)
```
>
> **参数**
>
> > - **eval_dataset** (paddlex.dataset): 评估数据集。
> > - **batch_size** (int): 验证数据批大小。默认为1。
> > - **return_details** (bool): 是否返回详细信息。默认为False。
>
> **返回值**
>
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当`return_details`为False时，返回metrics。 metrics为dict，包含键值：'acc1'、'acc5'，分别表示最大值的accuracy、前5个最大值的accuracy。eval_details为dict。包含键值：'true_labels'、'pred_scores'，分别代表真实类别id、每个类别的预测得分。

### <h3 id="13">predict</h3>

```python
predict(self, img_file, transforms=None, topk=1)
```

> 分类模型预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在`ResNet50.test_transforms`和`ResNet50.eval_transforms`中。如未在训练时定义eval_dataset，那在调用预测`predict`接口时，用户需要再重新定义test_transforms传入给`predict`接口。

> **参数**
>
> > - **img_file** (List[np.ndarray or str], str or np.ndarray): 预测图像或包含多张预测图像的列表，预测图像可以是路径或numpy数组(HWC排列，BGR格式)。
> > - **transforms** (paddlex.transforms): 数据预处理操作。
> > - **topk** (int): 预测时前k个最大值。

> **返回值**
>
> > - **dict** ｜ **List[dict]**: 如果输入为单张图像，返回dict。包含的键值为'category_id'、'category'、'score'，分别对应预测类别id、预测类别标签、预测得分。如果输入为多张图像，返回由每张图像预测结果组成的列表。

### <h3 id="14">analyze_sensitivity</h3>

```python
analyze_sensitivity(self, dataset, batch_size=8, criterion='l1_norm', save_dir='output')
```

> 模型敏感度分析接口。

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

> 模型剪裁接口。

> **参数**
> > - **pruned_flops** (float): 每秒浮点数运算次数（FLOPs）的剪裁比例。
> > - **save_dir** (None or str): 剪裁后模型保存路径。如果为None，剪裁完成后不会对模型进行保存。默认为None。

### <h3 id="16">quant_aware_train</h3>

```python
quant_aware_train(self, num_epochs, train_dataset, train_batch_size=64, eval_dataset=None, optimizer=None, save_interval_epochs=1, log_interval_steps=10, save_dir='output', learning_rate=.000025, warmup_steps=0, warmup_start_lr=0.0, lr_decay_epochs=(30, 60, 90), lr_decay_gamma=0.1, early_stop=False, early_stop_patience=5, use_vdl=True, quant_config=None)
```

> 分类模型在线量化接口。

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
> > - **learning_rate** (float): 默认优化器的初始学习率。默认为0.000025。
> > - **warmup_steps** (int): 默认优化器的warmup步数，学习率将在设定的步数内，从warmup_start_lr线性增长至设定的learning_rate，默认为0。
> > - **warmup_start_lr**(float): 默认优化器的warmup起始学习率，默认为0.0。
> > - **lr_decay_epochs** (list): 默认优化器的学习率衰减轮数。默认为[30, 60, 90]。
> > - **lr_decay_gamma** (float): 默认优化器的学习率衰减率。默认为0.1。
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


## <h2 id="2">其他分类模型</h2>


PaddleX提供了共计38种分类模型，所有分类模型均提供同`ResNet50`相同的训练`train`，评估`evaluate`，预测`predict`，敏感度分析`analyze_sensitivity`，剪裁`prune`和在线量化`quant_aware_train`接口，各模型效果可参考[模型库](../../appendix/model_zoo.md)。

| 模型              | 接口                    |
| :---------------- | :---------------------- |
| ResNet18          | paddlex.cls.ResNet18(num_classes=1000) |
| ResNet18_vd       | paddlex.cls.ResNet18_vd(num_classes=1000) |
| ResNet34          | paddlex.cls.ResNet34(num_classes=1000) |
| ResNet34_vd          | paddlex.cls.ResNet34_vd(num_classes=1000) |
| ResNet50          | paddlex.cls.ResNet50(num_classes=1000) |
| ResNet50_vd       | paddlex.cls.ResNet50_vd(num_classes=1000) |
| ResNet50_vd_ssld    | paddlex.cls.ResNet50_vd_ssld(num_classes=1000) |
| ResNet101          | paddlex.cls.ResNet101(num_classes=1000) |
| ResNet101_vd        | paddlex.cls.ResNet101_vd(num_classes=1000) |
| ResNet101_vd_ssld   | paddlex.cls.ResNet101_vd_ssld(num_classes=1000) |
| ResNet152 | paddlex.cls.ResNet152(num_classes=1000) |
| ResNet152_vd | paddlex.cls.ResNet152_vd(num_classes=1000) |
| ResNet200_vd | paddlex.cls.ResNet200_vd(num_classes=1000) |
| DarkNet53      | paddlex.cls.DarkNet53(num_classes=1000) |
| MobileNetV1         | paddlex.cls.MobileNetV1(num_classes=1000, scale=1.0) |
| MobileNetV2       | paddlex.cls.MobileNetV2(num_classes=1000, scale=1.0) |
| MobileNetV3_small       | paddlex.cls.MobileNetV3_small(num_classes=1000, scale=1.0) |
| MobileNetV3_small_ssld  | paddlex.cls.MobileNetV3_small_ssld(num_classes=1000, scale=1.0) |
| MobileNetV3_large   | paddlex.cls.MobileNetV3_large(num_classes=1000, scale=1.0) |
| MobileNetV3_large_ssld | paddlex.cls.MobileNetV3_large_ssld(num_classes=1000) |
| Xception41     | paddlex.cls.Xception41(num_classes=1000) |
| Xception65     | paddlex.cls.Xception65(num_classes=1000) |
| Xception71     | paddlex.cls.Xception71(num_classes=1000) |
| ShuffleNetV2     | paddlex.cls.ShuffleNetV2(num_classes=1000, scale=1.0) |
| ShuffleNetV2_swish     | paddlex.cls.ShuffleNetV2_swish(num_classes=1000) |
| DenseNet121      | paddlex.cls.DenseNet121(num_classes=1000) |
| DenseNet161       | paddlex.cls.DenseNet161(num_classes=1000) |
| DenseNet169       | paddlex.cls.DenseNet169(num_classes=1000) |
| DenseNet201       | paddlex.cls.DenseNet201(num_classes=1000) |
| DenseNet264       | paddlex.cls.DenseNet264(num_classes=1000) |
| HRNet_W18_C       | paddlex.cls.HRNet_W18_C(num_classes=1000) |
| HRNet_W30_C       | paddlex.cls.HRNet_W30_C(num_classes=1000) |
| HRNet_W32_C       | paddlex.cls.HRNet_W32_C(num_classes=1000) |
| HRNet_W40_C       | paddlex.cls.HRNet_W40_C(num_classes=1000) |
| HRNet_W44_C       | paddlex.cls.HRNet_W44_C(num_classes=1000) |
| HRNet_W48_C       | paddlex.cls.HRNet_W48_C(num_classes=1000) |
| HRNet_W64_C       | paddlex.cls.HRNet_W64_C(num_classes=1000) |
| AlexNet         | paddlex.cls.AlexNet(num_classes=1000) |
