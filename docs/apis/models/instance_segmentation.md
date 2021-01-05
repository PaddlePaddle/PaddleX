# Instance Segmentation

## MaskRCNN

```python
paddlex.det. MaskRCNN(num_classes=81, backbone='ResNet50', with_fpn=True, aspect_ratios=[0.5, 1.0, 2.0], anchor_sizes=[32, 64, 128, 256, 512], with_dcn=False, rpn_cls_loss='SigmoidCrossEntropy', rpn_focal_loss_alpha=0.25, rpn_focal_loss_gamma=2, rcnn_bbox_loss='SmoothL1Loss', rcnn_nms='MultiClassNMS', keep_top_k=100, nms_threshold=0.5, score_threshold=0.05, softnms_sigma=0.5, bbox_assigner='BBoxAssigner', fpn_num_channels=256, input_channel=3, rpn_batch_size_per_im=256, rpn_fg_fraction=0.5, test_pre_nms_top_n=None, test_post_nms_top_n=1000)
```

> Build a MaskRCNN detector. **Note that num_classes needs to be set to number of classes+background class in MaskRCNN. If an object includes humans and dogs, set num_classes to 3 so that the background class is included**

> **Parameters**

> > - **num_classes** (int): Number of classes including the background class. It is 81 by default.
> - > **backbone** (str): MaskRCNN backbone network in a value range of ['ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd', 'HRNet_W18', 'ResNet50_vd_ssld']. It is 'ResNet50' by default.
- **with_fpn** (bool): Whether to use FPN structure. It is true by default.
- **aspect_ratios** (list): Optional value of the anchor aspect ratio. It is [0.5, 1.0, 2.0] by default.
- **anchor_sizes** (list): Optional value of the anchor size. It is [32, 64, 128, 256, 512] by default.
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




#### train

```python
train(self, num_epochs, train_dataset, train_batch_size=1, eval_dataset=None, save_interval_epochs=1, log_interval_steps=20, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=1.0/800, warmup_steps=500, warmup_start_lr=1.0 / 2400, lr_decay_epochs=[8, 11], lr_decay_gamma=0.1, metric=None, use_vdl=False, early_stop=False, early_stop_patience=5, resume_checkpoint=None)
```

> MaskRCNN model training API. The function has a built-in `piecewise` learning rate attenuation policy and a `momentum` optimizer.

> **Parameters**
> - **num_epochs** (int): Number of training iteration epochs.
- **train_dataset** (paddlex.datasets): Training data reader.
- **train_batch_size** (int): Training data batch size. Currently, the detection supports only the single-card evaluation. The quotient of the training data batch size and the GPU quantity is a validation data batch size. It is 1 by default.
- **eval_dataset** (paddlex.datasets): Validation data reader.
- **save_interval_epochs** (int): Model saving interval (unit: number of iteration epochs). It is 1 by default.
- **log_interval_steps** (int): Training log output interval (unit: number of iterations). It is 2 by default.
- **save_dir** (str): Path where models are saved. It is 'output' by default.
- **pretrain_weights** (str): If it is a path, a pre-training model under the path is loaded. If it is a string 'IMAGENET', a model weight pre-trained on ImageNet image data is automatically downloaded. If it is a string 'COCO', a model weight pre-trained on the COCO dataset is automatically downloaded (Note: A COCO pre-training model for ResNet18 and HRNet_W18 is unavailable temporarily. If it is none, no pre-training model is used. It is None by default.
- **optimizer** (paddle.fluid.optimizer): Optimizer. When this parameter is none, a default optimizer is used: fluid.layers.piecewise_decay attenuation policy, fluid.optimizer. Momentum optimization method.
- **learning_rate** (float): Initial learning rate of the default optimizer. It is 0.00125 by default.
- **warmup_steps** (int): Number of steps to perform the warmup process by the default optimizer. It is 500 by default.
- **warmup_start_lr** (int): Initial learning rate of warmup of the default optimizer. It is 1.0/2400 by default.
- **lr_decay_epochs** (list): Number of learning rate attenuation epochs of the default optimizer. It is 8, 11[ by default].
- **lr_decay_gamma** (float): Attenuation rate of learning rate of the default optimizer. It is 0.1 by default.
- **metric** (bool): Evaluation method during training in the value range of 'COCO', 'VOC' .[It is None by default.]
- **use_vdl** (bool): Whether to use VisualDL for visualization. It is false by default.
- **early_stop** (float): Whether to use a policy for early termination of training. It is false by default.
- **early_stop_patience** (int): When a policy for early termination of training is used, training is terminated if the validation set precision continuously decreases or remains unchanged within early_stop_patience epochs. It is 5 by default.``
- **resume_checkpoint** (str): When training is resumed, specify a model path saved during the last training. If it is None, training is not resumed. It is None by default.



#### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, metric=None, return_details=False)
```

> MaskRCNN model evaluation API. The index box_mmap (when metric is set to COCO) on the validation set and the corresponding seg_mmap are returned after the model is evaluated.

> **Parameters**
> - **eval_dataset** (paddlex.datasets): Validation data reader.
- **batch_size** (int): Validation data batch size. It is 1 by default. Currently, it must be set to 1.
- **epoch_id** (int): Number of training epochs of the current evaluation model.
- **metric** (bool): Evaluation method during training in the value range of 'COCO', 'VOC' .[It is none by default. It is automatically selected according to the dataset passed by you. If it is VOCDetection, ]metric is 'VOC'. If it is COCODetection, metric is 'COCO'.````
- **return_details** (bool): Whether to return detailed information. It is false by default.


**Returned value**
> - **tuple** (metrics, eval_details) | **dict** (metrics): When return_details` is true, (metrics, eval_details) is returned. When return_details is false, metrics is returned.`metrics is dict and contains keywords: ' bbox_mmap' and 'segm_mmap' or ’bbox_map‘ and 'segm_map' which respectively indicates that the results of the average value of average accuracy rates of the predicted box and the segmented area under each threshold take the results of the average value (mmAP) and the average value of average accuracy rates (mAP). eval_details is dict and contains two keywords: ' bbox', 'mask' and ’gt‘. The key value of the keyword 'bbox' is a list of results of an predictred box. Each element in the list represents an prediction result. An prediction result is a list consisting of an image ID, an predictred box class ID, predicted box coordinates and an predicted box score. The key value of the keyword 'mask' is a list of results of an predicted area. Each element in the list represents an prediction result. An prediction result is a list consisting of an image ID, an predicted area class ID, predicted area coordinates and an predicted area score. The key value of the keyword ’gt‘ is information on the true annotated box and area.



#### predict

```python
predict(self, img_file, transforms=None)
```

> MaskRCNN model prediction API. Note that the image processing flow during prediction can be saved in `MaskRCNN.test_transforms and `MaskRCNN.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `predict API for prediction is called, you need to redefine and pass test_transforms to the predict API.````

> **Parameters**
> - **img_file** (str|np.ndarray): Path or numpy array of the predicted image (HWC arrangement, BGR format).
- **transforms** (paddlex.det.transforms): Data preprocessing operation.

**Returned value**
> - **list**: List of prediction results. Each element in the list has a dict. The key includes 'bbox', 'mask', 'category', 'category_id' and 'score' which indicate the box coordinate information, mask information, class, class ID and confidence of each predicted object respectively. The box coordinate information is [xmin, ymin, w, h], i.e. the x and y coordinates and the box width and height in the top left corner. The mask information is a binary image which has the same size as the original figure. The value 1 indicates that pixels belong to the prediction class. The value 0 indicates that pixels are a background.




#### batch_predict

```python
batch_predict(self, img_file_list, transforms=None)
```

> MaskRCNN model batch prediction API. Note that the image processing flow during prediction can be saved in `MaskRCNN.test_transforms and `MaskRCNN.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `batch_predict API for prediction is called, you need to redefine and pass test_transforms to the batch_predict API.````

> **Parameters**
> - **img_file_list** (list|tuple): Images in the list (or tuple) are simultaneously predicted. Elements in the list are predicted image paths or numpy arrays (HWC arrangement, BGR format).
- **transforms** (paddlex.det.transforms): Data preprocessing operation.

**Returned value**
> - **list**: Each element is a list which indicates prediction results of each image. Each element in the list of prediction results of each image has a dict and contains keywords: ' bbox', 'mask', 'category', 'category_id' and 'score' which indicate the box coordinate information, mask information, class, class ID and confidence of each predicted object respectively. The box coordinate information is xmin, ymin, w, h[, i.e. the x and y coordinates and the box width and height in the top left corner.]The mask information is a binary image which has the same size as the original figure. The value 1 indicates that pixels belong to the prediction class. The value 0 indicates that pixels are a background.
