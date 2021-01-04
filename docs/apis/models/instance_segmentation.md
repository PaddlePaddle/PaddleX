# Instance Segmentation

## MaskRCNN

```python
paddlex.det.MaskRCNN(num_classes=81, backbone='ResNet50', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[32, 64, 128, 256, 512], with_dcn=False, rpn_cls_loss='SigmoidCrossEntropy', rpn_focal_loss_alpha=0.25, rpn_focal_loss_gamma=2, rcnn_bbox_loss='SmoothL1Loss', rcnn_nms='MultiClassNMS', keep_top_k=100, nms_threshold=0.5, score_threshold=0.05, softnms_sigma=0.5, bbox_assigner='BBoxAssigner', fpn_num_channels=256, input_channel=3, rpn_batch_size_per_im=256, rpn_fg_fraction=0.5, test_pre_nms_top_n=None, test_post_nms_top_n=1000)

```

> 构建MaskRCNN检测器。**注意在MaskRCNN中，num_classes需要设置为类别数+背景类，如目标包括human、dog两种，则num_classes需设为3，多的一种为背景background类别**

> **参数**

> > - **num_classes** (int): 包含了背景类的类别数。默认为81。
> > - **backbone** (str): MaskRCNN的backbone网络，取值范围为['ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd', 'HRNet_W18', 'ResNet50_vd_ssld']。默认为'ResNet50'。
> > - **with_fpn** (bool): 是否使用FPN结构。默认为True。
> > - **aspect_ratios** (list): 生成anchor高宽比的可选值。默认为[0.5, 1.0, 2.0]。
> > - **anchor_sizes** (list): 生成anchor大小的可选值。默认为[32, 64, 128, 256, 512]。
> > - **with_dcn** (bool): backbone网络中是否使用deformable convolution network v2。默认为False。
> > - **rpn_cls_loss** (str): RPN部分的分类损失函数，取值范围为['SigmoidCrossEntropy', 'SigmoidFocalLoss']。当遇到模型误检了很多背景区域时，可以考虑使用'SigmoidFocalLoss'，并调整适合的`rpn_focal_loss_alpha`和`rpn_focal_loss_gamma`。默认为'SigmoidCrossEntropy'。
> > - **rpn_focal_loss_alpha** (float)：当RPN的分类损失函数设置为'SigmoidFocalLoss'时，用于调整正样本和负样本的比例因子，默认为0.25。当PN的分类损失函数设置为'SigmoidCrossEntropy'时，`rpn_focal_loss_alpha`的设置不生效。
> > - **rpn_focal_loss_gamma** (float): 当RPN的分类损失函数设置为'SigmoidFocalLoss'时，用于调整易分样本和难分样本的比例因子，默认为2。当RPN的分类损失函数设置为'SigmoidCrossEntropy'时，`rpn_focal_loss_gamma`的设置不生效。
> > - **rcnn_bbox_loss** (str): RCNN部分的位置回归损失函数，取值范围为['SmoothL1Loss', 'CIoULoss']。默认为'SmoothL1Loss'。
> > - **rcnn_nms** (str): RCNN部分的非极大值抑制的计算方法，取值范围为['MultiClassNMS', 'MultiClassSoftNMS','MultiClassCiouNMS']。默认为'MultiClassNMS'。当选择'MultiClassNMS'时，可以将`keep_top_k`设置成100、`nms_threshold`设置成0.5、`score_threshold`设置成0.05。当选择'MultiClassSoftNMS'时，可以将`keep_top_k`设置为300、`score_threshold`设置为0.01、`softnms_sigma`设置为0.5。当选择'MultiClassCiouNMS'时，可以将`keep_top_k`设置为100、`score_threshold`设置成0.05、`nms_threshold`设置成0.5。
> > - **keep_top_k** (int): RCNN部分在进行非极大值抑制计算后，每张图像保留最多保存`keep_top_k`个检测框。默认为100。
> > - **nms_threshold** (float): RCNN部分在进行非极大值抑制时，用于剔除检测框所需的IoU阈值。当`rcnn_nms`设置为`MultiClassSoftNMS`时，`nms_threshold`的设置不生效。默认为0.5。
> > - **score_threshold** (float): RCNN部分在进行非极大值抑制前，用于过滤掉低置信度边界框所需的置信度阈值。默认为0.05。
> > - **softnms_sigma** (float): 当`rcnn_nms`设置为`MultiClassSoftNMS`时，用于调整被抑制的检测框的置信度，调整公式为`score = score * weights, weights = exp(-(iou * iou) / softnms_sigma)`。默认设为0.5。
> > - **bbox_assigner** (str): 训练阶段，RCNN部分生成正负样本的采样方式。可选范围为['BBoxAssigner', 'LibraBBoxAssigner']。当目标物体的区域只占原始图像的一小部分时，可以考虑采用[LibraRCNN](https://arxiv.org/abs/1904.02701)中提出的IoU-balanced Sampling采样方式来获取更多的难分负样本，设置为'LibraBBoxAssigner'即可。默认为'BBoxAssigner'。
> > - **fpn_num_channels** (int): FPN部分特征层的通道数量。默认为256。
> > - **input_channel** (int): 输入图像的通道数量。默认为3。
> > - **rpn_batch_size_per_im** (int): 训练阶段，RPN部分每张图片的正负样本的数量总和。默认为256。
> > - **rpn_fg_fraction** (float): 训练阶段，RPN部分每张图片的正负样本数量总和中正样本的占比。默认为0.5。
> > - **test_pre_nms_top_n** (int)：预测阶段，RPN部分做非极大值抑制计算的候选框的数量。若设置为None, 有FPN结构的话，`test_pre_nms_top_n`会被设置成6000, 无FPN结构的话，`test_pre_nms_top_n`会被设置成1000。默认为None。
> > - **test_post_nms_top_n** (int): 预测阶段，RPN部分做完非极大值抑制后保留的候选框的数量。默认为1000。



#### train

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
> > - **pretrain_weights** (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为字符串'COCO'，则自动下载在COCO数据集上预训练的模型权重（注意：暂未提供ResNet18和HRNet_W18的COCO预训练模型）；若为None，则不使用预训练模型。默认为None。
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

#### evaluate

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
> > - **tuple** (metrics, eval_details) | **dict** (metrics): 当`return_details`为True时，返回(metrics, eval_details)，当return_details为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'和'segm_mmap'或者’bbox_map‘和'segm_map'，分别表示预测框和分割区域平均准确率平均值在各个IoU阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。eval_details为dict，包含`bbox`、`mask`和`gt`三个关键字。其中关键字`bbox`的键值是一个列表，列表中每个元素代表一个预测结果，一个预测结果是一个由图像id，预测框类别id, 预测框坐标，预测框得分组成的列表。关键字`mask`的键值是一个列表，列表中每个元素代表各预测框内物体的分割结果，分割结果由图像id、预测框类别id、表示预测框内各像素点是否属于物体的二值图、预测框得分。而关键字gt的键值是真实标注框的相关信息。

#### predict

```python
predict(self, img_file, transforms=None)
```

> MaskRCNN模型预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在`MaskRCNN.test_transforms`和`MaskRCNN.eval_transforms`中。如未在训练时定义eval_dataset，那在调用预测`predict`接口时，用户需要再重新定义test_transforms传入给`predict`接口。

> **参数**
>
> > - **img_file** (str|np.ndarray): 预测图像路径或numpy数组(HWC排列，BGR格式)。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值**
>
> > - **list**: 预测结果列表，列表中每个元素均为一个dict，key'bbox', 'mask', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、Mask信息，类别、类别id、置信度。其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。Mask信息为原图大小的二值图，1表示像素点属于预测类别，0表示像素点是背景。


#### batch_predict

```python
batch_predict(self, img_file_list, transforms=None)
```

> MaskRCNN模型批量预测接口。需要注意的是，只有在训练过程中定义了eval_dataset，模型在保存时才会将预测时的图像处理流程保存在`MaskRCNN.test_transforms`和`MaskRCNN.eval_transforms`中。如未在训练时定义eval_dataset，那在调用预测`batch_predict`接口时，用户需要再重新定义test_transforms传入给`batch_predict`接口。

> **参数**
>
> > - **img_file_list** (list|tuple): 对列表（或元组）中的图像同时进行预测，列表中的元素可以是预测图像路径或numpy数组(HWC排列，BGR格式)。
> > - **transforms** (paddlex.det.transforms): 数据预处理操作。
>
> **返回值**
>
> > - **list**: 每个元素都为列表，表示各图像的预测结果。在各图像的预测结果列表中，每个元素均为一个dict，包含关键字：'bbox', 'mask', 'category', 'category_id', 'score'，分别表示每个预测目标的框坐标信息、Mask信息，类别、类别id、置信度。其中框坐标信息为[xmin, ymin, w, h]，即左上角x, y坐标和框的宽和高。Mask信息为原图大小的二值图，1表示像素点属于预测类别，0表示像素点是背景。
