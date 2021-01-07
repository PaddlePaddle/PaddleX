# Object Detection

## paddlex.det. PPYOLO

```python
paddlex.det. PPYOLO(num_classes=80, backbone='ResNet50_vd_ssld', with_dcn_v2=True, anchors=None, anchor_masks=None, use_coord_conv=True, use_iou_aware=True, use_spp=True, use_drop_block=True, scale_x_y=1.05, ignore_threshold=0.7, label_smooth=False, use_iou_loss=True, use_matrix_nms=True, nms_score_threshold=0.01, nms_topk=1000, nms_keep_topk=100, nms_iou_threshold=0.45, train_random_shapes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
```

> Build a PPYOLO detector. **Note that num_classes does not need to include background class in PPYOLO. If an object includes humans and dogs, set num_classes to 2, which is different from FasterRCNN/MaskRCNN here**

> **Parameters**
> > - **num_classes** (int): Number of classes. It is 80 by default.
> > - **backbone** (str): PPYOLO backbone network in a value range of 'ResNet50_vd_ssld' .[It is 'ResNet50_vd_ssld' by default.]
> > - **with_dcn_v2** (bool): Whether Backbone uses DCNv2 structure. It is true by default.
> > - **anchors** (list|tuple): Width and height of the anchor box. When it is none, it indicates using the default
> >                  [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
>                   [59, 119], [116, 90], [156, 198], [373, 326]].
> > - **anchor_masks** (list|tuple): When the PPYOLO loss is calculated, the mask index of the anchor is used. When it is none, it indicates using the default
> >                    [[6, 7, 8], [3, 4, 5], [0, 1, 2]].
> > - **use_coord_conv** (bool): Whether to use CoordConv. It is true by default.
> > - **use_iou_aware** (bool): Whether to use an IoU Aware branch. It is true by default.
> > - **use_spp** (bool): Whether to use Spatial Pyramid Pooling structure. It is true by default.
> > - **use_drop_block** (bool): Whether to use Drop Block. It is true by default.
> > - **scale_x_y** (float): Factor when the center is adjusted. It is 1.05 by default.
> > - **use_iou_loss** (bool): Whether to use IoU loss. It is true by default.
> > - **use_matrix_nms** (bool): Whether to use Matrix NMS. It is true by default.
> > - **ignore_threshold** (float): When the PPYOLO loss is calculated, the confidence of predicted boxes of which the IoU is greater than `ignore_threshold` is ignored. It is 0.7 by default.
> > - **nms_score_threshold** (float): Confidence score threshold of the detected box. Any box of which the confidence is smaller than the threshold shall be ignored. It is 0.01 by default.
> > - **nms_topk** (int): Maximum number of detected boxes reserved according to the confidence when NMS is performed. It is 1000 by default.
> > - **nms_keep_topk** (int): Total number of detected boxes to be reserved for each image after NMS is performed. It is 100 by default.
> > - **nms_iou_threshold** (float): IOU threshold used to eliminate detected boxes when NMS is performed. It is 0.45 by default.
> > - **label_smooth** (bool): Whether to use label smooth. It is false by default.
> > - **train_random_shapes** (list|tuple): Image size randomly selected from the list during training. It is 320, 352, 384, 416, 448, 480, 512, 544, 576, 608[ by default].



### train

```python
train(self, num_epochs, train_dataset, train_batch_size=8, eval_dataset=None, save_interval_epochs=20, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=1.0/8000, warmup_steps=1000, warmup_start_lr=0.0, lr_decay_epochs=[213, 240], lr_decay_gamma=0.1, metric=None, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5, resume_checkpoint=None, use_ema=True, ema_decay=0.9998)
```

> PPYOLO model training API. The function has a built-in `piecewise` learning rate attenuation policy and a `momentum` optimizer.

> **Parameters**
>
> > - **num_epochs** (int): Number of training iteration epochs.
> > -**train_dataset** (paddlex.datasets): Training data reader.
> > -**train_batch_size** (int): Training data batch size. Currently, the detection supports only the single-card evaluation. The quotient of the training data batch size and the GPU quantity is a validation data batch size. It is 8 by default.
> > - **eval_dataset** (paddlex.datasets): Validation data reader.
> > -**save_interval_epochs** (int): Model saving interval (unit: number of iteration epochs). It is 20 by default.
> > -**log_interval_steps** (int): Training log output interval (unit: number of iterations). It is 2 by default.
> > -**save_dir** (str): Path where models are saved. It is 'output' by default.
> > -**pretrain_weights** (str): If it is a path, a pre-training model under the path is loaded. If it is a string 'IMAGENET', a model weight pre-trained on ImageNet image data is automatically downloaded. If it is a string 'COCO', a model weight pre-trained on the COCO dataset is automatically downloaded. If it is none, no pre-training model is used. It is 'IMAGENET' by default.
> > -**optimizer** (paddle.fluid.optimizer): Optimizer. When this parameter is none, a default optimizer is used: fluid.layers.piecewise_decay attenuation policy, fluid.optimizer. Momentum optimization method.
> > -**learning_rate** (float): Learning rate of the default optimizer. It is 1.0/8000 by default.
> > -**warmup_steps** (int): Number of steps to perform the warmup process by the default optimizer. It is 1000 by default.
> > -**warmup_start_lr** (int): Initial learning rate of warmup of the default optimizer. It is 0.0 by default.
> > -**lr_decay_epochs** (list): Number of learning rate attenuation epochs of the default optimizer. It is [213, 240] by default.
> > -**lr_decay_gamma** (float): Attenuation rate of learning rate of the default optimizer. It is 0.1 by default.
> > -**metric** (bool): Evaluation method during training in the value range of ['COCO', 'VOC'] . It is None by default.
> > -**use_vdl** (bool): Whether to use VisualDL for visualization. It is false by default.
> > -**sensitivities_file** (str): If it is a path, sensitivity information under the path is loaded to perform pruning. If it is a string 'DEFAULT', sensitivity information obtained on PascalVOC data is automatically downloaded to perform pruning. If it is none, no pruning is performed. It is None by default.
> > -**eval_metric_loss** (float): Tolerable precision loss. It is 0.05 by default.
> > -**early_stop** (bool): Whether to use a policy for early termination of training. It is false by default.
> > -**early_stop_patience** (int): When a policy for early termination of training is used, training is terminated if the validation set precision continuously decreases or remains unchanged within `early_stop_patience` epochs. It is 5 by default.
> > -**resume_checkpoint** (str): When training is resumed, specify a model path saved during the last training. If it is None, training is not resumed. It is None by default.
> > -**use_ema** (bool): Whether to use exponential attenuation to calculate a parameter sliding average value. It is true by default.
> > -**ema_decay** (float): Exponential attenuation rate. It is 0.9998 by default.



### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```

> PPYOLO model evaluation API. The index `box_map` (when metric is set to 'VOC') or `box_mmap` (when metric is set to `COCO`) on the validation set is returned after the model is evaluated.

> **Parameters**
> >  - **eval_dataset** (paddlex.datasets): Validation data reader.
> > - **batch_size** (int): Validation data batch size. It is 1 by default.
> > - **epoch_id** (int): Number of training epochs of the current evaluation model.
> > - **metric** (bool): Evaluation method during training in the value range of ['COCO', 'VOC']. It is none by default. It is automatically selected according to the dataset passed by you. If it is VOCDetection, `metric` is 'VOC'. If it is COCODetection, `metric` is 'COCO'. If it is a EasyData dataset, 'VOC' is also used.
> > - **return_details** (bool): Whether to return detailed information. It is false by default.


**Returned value**
> - **tuple** (metrics, eval_details) | **dict** (metrics): When `return_details` is true, (metrics, eval_details) is returned. When `return_details` is false, metrics is returned. metrics is dict and contains keywords: '`bbox_mmap' or `bbox_map` which respectively indicates that the results of the average value of average accuracy rates under each threshold take the results of the average value (mmAP) and the average value of average accuracy rates (mAP). eval_details is dict and contains two keywords: 'bbox' and 'gt'. The key value of the keyword 'bbox' is a list. Each element in the list represents an prediction result. An prediction result is a list consisting of an image ID, an predicted box class ID, predicted box coordinates and an predicted box score. The key value of the keyword 'gt' is information on the true annotated box.



### predict

```python
predict(self, img_file, transforms=None)
```

> PPYOLO model prediction API. Note that the image processing flow during prediction can be saved in `YOLOv3.test_transforms' and `YOLOv3.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `predict' API for prediction is called, you need to redefine and pass `test_transforms` to the predict API

> **Parameters**
>
>> - **img_file** (str|np.ndarray): Path or numpy array of the predicted image (HWC arrangement, BGR format).
>>- **transforms** (paddlex.det.transforms): Data preprocessing operation.

>**Returned value**
>
> >- **list**: List of prediction results. Each element in the list has a dict. The key includes 'bbox', 'category', 'category_id' and 'score' which indicate the box coordinate information, class, class ID and confidence of each predicted object respectively. The box coordinate information is [xmin, ymin, w, h], i.e. the x and y coordinates and the box width and height in the top left corner.




### batch_predict

```python
batch_predict(self, img_file_list, transforms=None)
```

> PPYOLO model batch prediction API. Note that the image processing flow during prediction can be saved in `YOLOv3.test_transforms` and `YOLOv3.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `batch_predict` API for prediction is called, you need to redefine and pass `test_transforms` to the batch_predict API

> **Parameters**
>
>> - **img_file_list** (str|np.ndarray): Images in the list (or tuple) are simultaneously predicted. Elements in the list are predicted image paths or numpy arrays (HWC arrangement, BGR format).
>>- **transforms** (paddlex.det.transforms): Data preprocessing operation.

>**Returned value**
>
>> - **list**: Each element is a list which indicates prediction results of each image. Each element in the list of prediction results of each image has a dict. The key includes 'bbox', 'category', 'category_id' and 'score' which indicate the box coordinate information, class, class ID and confidence of each predicted object respectively. The box coordinate information is [xmin, ymin, w, h], i.e. the x and y coordinates and the box width and height in the top left corner.


## paddlex.det. YOLOv3

```python
paddlex.det. YOLOv3(num_classes=80, backbone='MobileNetV1', anchors=None, anchor_masks=None, ignore_threshold=0.7, nms_score_threshold=0.01, nms_topk=1000, nms_keep_topk=100, nms_iou_threshold=0.45, label_smooth=False, train_random_shapes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
```

> Build a YOLOv3 detector. **Note that num_classes does not need to include background class in YOLOv3. If an object includes humans and dogs, set num_classes to 2, which is different from FasterRCNN/MaskRCNN here**

> **Parameters**
> > -  **num_classes** (int): Number of classes. It is 80 by default.
> > - **backbone** (str): YOLOv3 backbone network in a value range of ['DarkNet53', 'ResNet34', 'MobileNetV1' and 'MobileNetV3_large' ]. It is 'MobileNetV1' by default.
> > -**anchors** (list|tuple): Width and height of the anchor box. When it is none, it indicates using the default
> >                  [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
>                   [59, 119], [116, 90], [156, 198], [373, 326]].
> > - **anchor_masks** (list|tuple): When the YOLOv3 loss is calculated, the mask index of the anchor is used. When it is none, it indicates using the default
> >                    [[6, 7, 8], [3, 4, 5], [0, 1, 2]].
> > -**ignore_threshold** (float): When the YOLOv3 loss is calculated, the confidence of predicted boxes of which the IoU is greater than `ignore_threshold` is ignored. It is 0.7 by default.
> > -**nms_score_threshold** (float): Confidence score threshold of the detected box. Any box of which the confidence is smaller than the threshold shall be ignored. It is 0.01 by default.
> > -**nms_topk** (int): Maximum number of detected boxes reserved according to the confidence when NMS is performed. It is 1000 by default.
> > -**nms_keep_topk** (int): Total number of detected boxes to be reserved for each image after NMS is performed. It is 100 by default.
> > -**nms_iou_threshold** (float): IOU threshold used to eliminate detected boxes when NMS is performed. It is 0.45 by default.
> > -**label_smooth** (bool): Whether to use label smooth. It is false by default.
> > -**train_random_shapes** (list|tuple): Image size randomly selected from the list during training. It is [320, 352, 384, 416, 448, 480, 512, 544, 576, 608] by default.



### train

```python
train(self, num_epochs, train_dataset, train_batch_size=8, eval_dataset=None, save_interval_epochs=20, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=1.0/8000, warmup_steps=1000, warmup_start_lr=0.0, lr_decay_epochs=[213, 240], lr_decay_gamma=0.1, metric=None, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5, resume_checkpoint=None)
```

> YOLOv3 model training API. The function has a built-in `piecewise` learning rate attenuation policy and a `momentum` optimizer.

> **Parameters**
> > -  **num_epochs** (int): Number of training iteration epochs.
> > - **train_dataset** (paddlex.datasets): Training data reader.
> > - **train_batch_size** (int): Training data batch size. Currently, the detection supports only the single-card evaluation. The quotient of the training data batch size and the GPU quantity is a validation data batch size. It is 8 by default.
> > - **eval_dataset** (paddlex.datasets): Validation data reader.
> > - **save_interval_epochs** (int): Model saving interval (unit: number of iteration epochs). It is 20 by default.
> > - **log_interval_steps** (int): Training log output interval (unit: number of iterations). It is 2 by default.
> > - **save_dir** (str): Path where models are saved. It is 'output' by default.
> > - **pretrain_weights** (str): If it is a path, a pre-training model under the path is loaded. If it is a string 'IMAGENET', a model weight pre-trained on ImageNet image data is automatically downloaded. If it is a string 'COCO', a model weight pre-trained on the COCO dataset is automatically downloaded. If it is none, no pre-training model is used. It is 'IMAGENET' by default.
> > - **optimizer** (paddle.fluid.optimizer): Optimizer. When this parameter is none, a default optimizer is used: fluid.layers.piecewise_decay attenuation policy, fluid.optimizer. Momentum optimization method.
> > - **learning_rate** (float): Learning rate of the default optimizer. It is 1.0/8000 by default.
> > - **warmup_steps** (int): Number of steps to perform the warmup process by the default optimizer. It is 1000 by default.
> > - **warmup_start_lr** (int): Initial learning rate of warmup of the default optimizer. It is 0.0 by default.
> > - **lr_decay_epochs** (list): Number of learning rate attenuation epochs of the default optimizer. It is [213, 240] by default.
> > - **lr_decay_gamma** (float): Attenuation rate of learning rate of the default optimizer. It is 0.1 by default.
> > - **metric** (bool): Evaluation method during training in the value range of ['COCO', 'VOC']. It is None by default.
> > - **use_vdl** (bool): Whether to use VisualDL for visualization. It is false by default.
> > - **sensitivities_file** (str): If it is a path, sensitivity information under the path is loaded to perform pruning. If it is a string 'DEFAULT', sensitivity information obtained on PascalVOC data is automatically downloaded to perform pruning. If it is none, no pruning is performed. It is None by default.
> > - **eval_metric_loss** (float): Tolerable precision loss. It is 0.05 by default.
> > - **early_stop** (bool): Whether to use a policy for early termination of training. It is false by default.
> > - **early_stop_patience** (int): When a policy for early termination of training is used, training is terminated if the validation set precision continuously decreases or remains unchanged within early_stop_patience epochs. It is 5 by default.``
> > - **resume_checkpoint** (str): When training is resumed, specify a model path saved during the last training. If it is None, training is not resumed. It is None by default.

### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```

> YOLOv3 model evaluation API. The index `box_map` (when metric is set to 'VOC') or `box_mmap` (when metric is set to `COCO`) on the validation set is returned after the model is evaluated.

> **Parameters**
>
> > - **eval_dataset** (paddlex.datasets): Validation data reader.
> > -**batch_size** (int): Validation data batch size. It is 1 by default.
> > -**epoch_id** (int): Number of training epochs of the current evaluation model.
> > -**metric** (bool): Evaluation method during training in the value range of ['COCO', 'VOC']. It is none by default. It is automatically selected according to the dataset passed by you. If it is VOCDetection, metric is ['VOC']. If it is COCODetection, metric is 'COCO'. If it is a EasyData dataset, 'VOC' is also used.
> > -**return_details** (bool): Whether to return detailed information. It is false by default.
>>
>**Returned value**
>
> >- **tuple** (metrics, eval_details) | **dict** (metrics): When `return_details` is true, (metrics, eval_details) is returned. When `return_details` is false, metrics is returned.`metrics is dict and contains keywords: 'bbox_mmap' or `bbox_map` which respectively indicates that the results of the average value of average accuracy rates under each threshold take the results of the average value (mmAP) and the average value of average accuracy rates (mAP). eval_details is dict and contains two keywords: 'bbox` and `gt`. The key value of the keyword `bbox`; is a list. Each element in the list represents an prediction result. An prediction result is a list consisting of an image ID, an predicted box class ID, predicted box coordinates and an predicted box score. The key value of the keyword `gt` is information on the true annotated box.



### predict

```python
predict(self, img_file, transforms=None)
```

> YOLOv3 model prediction API. Note that the image processing flow during prediction can be saved in `YOLOv3.test_transforms` and `YOLOv3.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `predict` API for prediction is called, you need to redefine and pass `test_transforms` to the predict API

> **Parameters**
>
> >- **img_file** (str|np.ndarray): Path or numpy array of the predicted image (HWC arrangement, BGR format).
> >- **transforms** (paddlex.det.transforms): Data preprocessing operation.

**Returned value**
> - **list**: List of prediction results. Each element in the list has a dict. The key includes 'bbox', 'category', 'category_id' and 'score' which indicate the box coordinate information, class, class ID and confidence of each predicted object respectively. The box coordinate information is [xmin, ymin, w, h], i.e. the x and y coordinates and the box width and height in the top left corner.




### batch_predict

```python
batch_predict(self, img_file_list, transforms=None)
```

> YOLOv3 model batch prediction API. Note that the image processing flow during prediction can be saved in `YOLOv3.test_transforms` and `YOLOv3.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `batch_predict` API for prediction is called, you need to redefine and pass `test_transforms` to the `batch_predict` API

> **Parameters**
>
>>- **img_file_list** (str|np.ndarray): Images in the list (or tuple) are simultaneously predicted. Elements in the list are predicted image paths or numpy arrays (HWC arrangement, BGR format).
>>- **transforms** (paddlex.det.transforms): Data preprocessing operation.

>**Returned value**
>
> - **list**: Each element is a list which indicates prediction results of each image. Each element in the list of prediction results of each image has a dict. The key includes 'bbox', 'category', 'category_id' and 'score' which indicate the box coordinate information, class, class ID and confidence of each predicted object respectively. The box coordinate information is [xmin, ymin, w, h], i.e. the x and y coordinates and the box width and height in the top left corner.





## paddlex.det. FasterRCNN

```python
paddlex.det. FasterRCNN(num_classes=81, backbone='ResNet50', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[32, 64, 128, 256, 512], with_dcn=False, rpn_cls_loss='SigmoidCrossEntropy', rpn_focal_loss_alpha=0.25, rpn_focal_loss_gamma=2, rcnn_bbox_loss='SmoothL1Loss', rcnn_nms='MultiClassNMS', keep_top_k=100, nms_threshold=0.5, score_threshold=0.05, softnms_sigma=0.5, bbox_assigner='BBoxAssigner', fpn_num_channels=256, input_channel=3, rpn_batch_size_per_im=256, rpn_fg_fraction=0.5, test_pre_nms_top_n=None, test_post_nms_top_n=1000)
```

> Build a FasterRCNN detector. **Note that num_classes needs to be set to number of classes+background class in FasterRCNN. If an object includes humans and dogs, set num_classes to 3 so that the background class is included**

> **Parameters**

> > - **num_classes** (int): Number of classes including the background class. It is 81 by default.
> > - **backbone** (str): FasterRCNN backbone network in a value range of ['ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd', 'HRNet_W18', 'ResNet50_vd_ssld']. It is 'ResNet50' by default.
> > -**with_fpn** (bool): Whether to use FPN structure. It is true by default.
> > -**aspect_ratios** (list): Optional value of the anchor aspect ratio. It is [0.5, 1.0, 2.0] by default.
> > -**anchor_sizes** (list): Optional value of the anchor size. It is [32, 64, 128, 256, 512] by default.
> > - **with_dcn** (bool): Whether to use deformable convolution network v2 in the backbone. Default: False.
> > - **rpn_cls_loss** (str): The classification loss function for RPN in a value range of ['SigmoidCrossEntropy', 'SigmoidFocalLoss']。When there are many false positives in backgorund areas, 'SigmoidFocalLoss' with appropriate `rpn_focal_loss_alpha` and `rpn_focal_loss_gamma` settings may be a better option. Default: 'SigmoidCrossEntropy'.
> > - **rpn_focal_loss_alpha** (float)：Hyper-parameter to balance the positive and negative examples where 'SigmoidFocalLoss' is set as the lassification loss function for RPN, Default: 0.25. If use 'SigmoidCrossEntropy', `rpn_focal_loss_alpha` has no effect.
> > - **rpn_focal_loss_gamma** (float): Hyper-parameter to balance the easy and hard examples where 'SigmoidFocalLoss' is set as the lassification loss function for RPN, Default: 2. If use 'SigmoidCrossEntropy', `rpn_focal_loss_gamma` has no effect.
> > - **rcnn_bbox_loss** (str): The location regression loss function for RCNN in a value range of ['SmoothL1Loss', 'CIoULoss']. Default: 'SmoothL1Loss'.
> > - **rcnn_nms** (str): The non-maximum suppression(NMS) method for RCNN, in a value range of ['MultiClassNMS', 'MultiClassSoftNMS','MultiClassCiouNMS']. Default: 'MultiClassNMS'. When 'MultiClassNMS' is set, `keep_top_k`, `nms_threshold` and `score_threshold` can be set as 100, 0.5 and 0.05 respectively. When 'MultiClassSoftNMS' is set, `keep_top_k`, `score_threshold` and `softnms_sigma` can be set as 300, 0.01 and 0.5 respectively. When 'MultiClassCiouNMS' is set, `keep_top_k`, `score_threshold` and `nms_threshold` can be set as 100, 0.05 and 0.5 respectively.
> > - **keep_top_k** (int): The Number of total bouning boxes to be kept per image after NMS step for RCNN. Default: 100.
> > - **nms_threshold** (float): The IoU threshold to filter out bounding boxes in NMS for RCNN. When `rcnn_nms` is set as `MultiClassSoftNMS`，`nms_threshold` has no effect。Default: 0.5.
> > - **score_threshold** (float): The confidence score threshold to filter out bounding boxes before nms. Default: 0.05.
> > - **softnms_sigma** (float): When `rcnn_nms` is set as `MultiClassSoftNMS`, `softnms_sigma` is used to adjust the confidence score of suppressed bounding boxes according to `score = score * weights, weights = exp(-(iou * iou) / softnms_sigma)`. Default: 0.5.
> > - **bbox_assigner** (str): The method of sampling positive and negative examples during the traing phase, in a value range of ['BBoxAssigner', 'LibraBBoxAssigner']. If the size of objects is a small portion of the image, [LibraRCNN](https://arxiv.org/abs/1904.02701) proposed a IoU-balanced sampling method to abtain more hard-negative examples, namely 'LibraBBoxAssigner'. Default: 'BBoxAssigner'.
> > - **fpn_num_channels** (int): The number of channels of feature maps in FPN2. Default: 56.
> > - **input_channel** (int): The number of channels of a input image. Default: 3.
> > - **rpn_batch_size_per_im** (int): Total number of training examples per image for RPN. Default: 256.
> > - **rpn_fg_fraction** (float): The fraction of positive examples in total train examples for RPN. Default: 0.5.
> > - **test_pre_nms_top_n** (int)：The number of predicted bounding boxes fed into NMS step. If set as None, `test_pre_nms_top_n` will be set as 6000 with a FPN or 1000 with no FPN. Default: None.
> > - **test_post_nms_top_n** (int): The number of predicted bounding boxes kept after NMS step. Default: 1000.



### train

```python
train(self, num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, save_interval_epochs=1, log_interval_steps=2,save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=0.0025, warmup_steps=500, warmup_start_lr=1.0/1200, lr_decay_epochs=[8, 11], lr_decay_gamma=0.1, metric=None, use_vdl=False, early_stop=False, early_stop_patience=5, resume_checkpoint=None)
```

> FasterRCNN model training API. The function has a built-in `piecewise` learning rate attenuation policy and a `momentum` optimizer.

> **Parameters**
>
> - **num_epochs** (int): Number of training iteration epochs.
> > - **train_dataset** (paddlex.datasets): Training data reader.
> > - **train_batch_size** (int): Training data batch size. Currently, the detection supports only the single-card evaluation. The quotient of the training data batch size and the GPU quantity is a validation data batch size. It is 2 by default.
> > - **eval_dataset** (paddlex.datasets): Validation data reader.
> > - **save_interval_epochs** (int): Model saving interval (unit: number of iteration epochs). It is 1 by default.
> > - **log_interval_steps** (int): Training log output interval (unit: number of iterations). It is 2 by default.
> > - **save_dir** (str): Path where models are saved. It is 'output' by default.
> > - **pretrain_weights** (str): If it is a path, a pre-training model under the path is loaded. If it is a string 'IMAGENET', a model weight pre-trained on ImageNet image data is automatically downloaded. If it is a string 'COCO', a model weight pre-trained on the COCO dataset is automatically downloaded (Note: A COCO pre-training model for ResNet18 is unavailable temporarily. If it is none, no pre-training model is used. It is 'IMAGENET' by default.
> > - **optimizer** (paddle.fluid.optimizer): Optimizer. When this parameter is none, a default optimizer is used: fluid.layers.piecewise_decay attenuation policy, fluid.optimizer. Momentum optimization method.
> > - **learning_rate** (float): Initial learning rate of the default optimizer. It is 0.0025 by default.
> > - **warmup_steps** (int): Number of steps to perform the warmup process by the default optimizer. It is 500 by default.
> > - **warmup_start_lr** (int): Initial learning rate of warmup of the default optimizer. It is 1.0/1200 by default.
> > - **lr_decay_epochs** (list): Number of learning rate attenuation epochs of the default optimizer. It is [8, 11] by default.
> > - **lr_decay_gamma** (float): Attenuation rate of learning rate of the default optimizer. It is 0.1 by default.
> > - **metric** (bool): Evaluation method during training in the value range of ['COCO', 'VOC']. It is None by default.
> > - **use_vdl** (bool): Whether to use VisualDL for visualization. It is false by default.
> > - **early_stop** (float): Whether to use a policy for early termination of training. It is false by default.
> > - **early_stop_patience** (int): When a policy for early termination of training is used, training is terminated if the validation set precision continuously decreases or remains unchanged within `early_stop_patience` epochs. It is 5 by default.
> > - **resume_checkpoint** (str): When training is resumed, specify a model path saved during the last training. If it is None, training is not resumed. It is None by default.



### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```

> FasterRCNN model evaluation API. The index box_map (when metric is set to `VOC`) or box_mmap (when metric is set to COCO) on the validation set is returned after the model is evaluated.

> **Parameters**
>
>> - **eval_dataset** (paddlex.datasets): Validation data reader.
>> - **batch_size** (int): Validation data batch size. It is 1 by default. Currently, it must be set to 1.
>> - **epoch_id** (int): Number of training epochs of the current evaluation model.
>> - **metric** (bool): Evaluation method during training in the value range of ['COCO', 'VOC']. It is none by default. It is automatically selected according to the dataset passed by you. If it is VOCDetection, 'metric' is 'VOC'. If it is COCODetection, 'metric' is 'COCO'.
>> - **return_details** (bool): Whether to return detailed information. It is false by default.


**Returned value**
> - **tuple** (metrics, eval_details) | **dict** (metrics): When 'return_details` is true, (metrics, eval_details) is returned. When 'return_details' is false, metrics is returned. metrics is dict and contains keywords: '`bbox_mmap' or 'bbox_map' which respectively indicates that the results of the average value of average accuracy rates under each threshold take the results of the average value (mmAP) and the average value of average accuracy rates (mAP). eval_details is dict and contains two keywords: ' bbox' and `gt`. The key value of the keyword `bbox`; is a list. Each element in the list represents an prediction result. An prediction result is a list consisting of an image ID, an predicted box class ID, predicted box coordinates and an predicted box score. The key value of the keyword `gt` is information on the true annotated box.



### predict

```python
predict(self, img_file, transforms=None)
```

> FasterRCNN model prediction API. Note that the image processing flow during prediction can be saved in `FasterRCNN.test_transforms' and `FasterRCNN.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `predict' API for prediction is called, you need to redefine and pass test_transforms to the 'predict' API.

> **Parameters**
> - **img_file** (str|np.ndarray): Path or numpy array of the predicted image (HWC arrangement, BGR format).
- **transforms** (paddlex.det.transforms): Data preprocessing operation.

**Returned value**
> - **list**: List of prediction results. Each element in the list has a dict. The key includes 'bbox', 'category', 'category_id' and 'score' which indicate the box coordinate information, class, class ID and confidence of each predicted object respectively. The box coordinate information is [xmin, ymin, w, h], i.e. the x and y coordinates and the box width and height in the top left corner.




### batch_predict

```python
batch_predict(self, img_file_list, transforms=None)
```

> FasterRCNN model batch prediction API. Note that the image processing flow during prediction can be saved in `FasterRCNN.test_transforms and `FasterRCNN.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `batch_predict` API for prediction is called, you need to redefine and pass test_transforms to the `batch_predict` API.

> **Parameters**
>
> >- **img_file_list** (list|tuple): Images in the list (or tuple) are simultaneously predictred. Elements in the list are predicted image paths or numpy arrays (HWC arrangement, BGR format).
>>- **transforms** (paddlex.det.transforms): Data preprocessing operation.

>**Returned value**
>
> >- **list**: Each element is a list which indicates prediction results of each image. Each element in the list of prediction results of each image has a dict. The key includes 'bbox', 'category', 'category_id' and 'score' which indicate the box coordinate information, class, class ID and confidence of each predicted object respectively. The box coordinate information is [xmin, ymin, w, h], i.e. the x and y coordinates and the box width and height in the top left corner.
