# Semantic Segmentation

## paddlex.seg.DeepLabv3p

```python
paddlex. seg.DeepLabv3p(num_classes=2, backbone='MobileNetV2_x1.0', output_stride=16, aspp_with_sep_conv=True, decoder_use_sep_conv=True, encoder_with_aspp=True, enable_decoder=True, use_bce_loss=False, use_dice_loss=False, class_weight=None, ignore_index=255, pooling_crop_size=None, input_channel=3)
```

> Build a DeepLabv3p segmenter.

> **Parameters**

> > - **num_classes** (int): Number of classes.
> - > **backbone** (str): DeepLabv3+ backbone network to implement the calculation of characteristic images in a value range of 'Xception65', 'Xception41', 'MobileNetV2_x0.25', 'MobileNetV2_x0.5', 'MobileNetV2_x1.0', 'MobileNetV2_x1.5', 'MobileNetV2_x2.0', 'MobileNetV3_large_x1_0_ssld'. It is 'MobileNetV2_x1.0' by default.[ ]
- **output_stride** (int): Downsampling multiple of the backbone output characteristic image relative to the input. It is generally 8 or 16. It is 16 by default.
- **aspp_with_sep_conv** (bool): Whether the decoder module uses separable convolutions. It is true by default.
- **decoder_use_sep_conv** (bool)：Whether the decoder module uses separable convolutions. It is true by default.
- **encoder_with_aspp** (bool): Whether to use an ASPP module in the encoder phase. It is true by default.
- **enable_decoder** (bool): Whether to use a decoder module. It is true by default.
- **use_bce_loss** (bool): Whether to use bce loss as a network loss function. The bce loss function can be used for two kinds of segmentation only and may be used with dice loss. It is false by default.
- **use_dice_loss** (bool): Whether to use dice loss as a network loss function. The dice loss function can be used for two kinds of segmentation only and may be used with dice loss. When both use_bce_loss` and use_dice_loss are false, the cross entropy loss function is used.`It is false by default.``
- **class_weight** (list/str): Weight of various losses of the cross entropy loss function. When class_weight` is a list, the length shall be `num_classes`.`When class_weight` is str, weight.`lower() shall be 'dynamic'. At this moment, the corresponding weight is automatically calculated according to the proportion of all classes of pixels in each round. The weight of each class is as follows: Proportion of each class * num_classes. When class_weight is the default none, the weight of each class is 1, i.e. the usually used cross entropy loss function.
- **ignore_index** (int): Value ignored on a label. A pixel of which the label is ignore_index does not participate in the calculation of the loss function. It is 255 by default.``
- **pooling_crop_size** (int): When backbone is MobileNetV3_large_x1_0_ssld`, this parameter must be set to a model input size during training in `W, H[ format]. For example, if the model input size is [512, 512], `pooling_crop_size` shall be set to 512, 512[.]This parameter is used when an image average is obtained in the encoder module. If it is none, an average is directly calculated. If it is a model input size, an average is obtained using the avg_pool` operator.`It is None by default.
- **input_channel** (int): Number of input image channels. It is 3 by default.



### train

```python
train(self, num_epochs, train_dataset, train_batch_size=2, eval_dataset=None, eval_batch_size=1, save_interval_epochs=1, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=0.01, lr_decay_power=0.9, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5, resume_checkpoint=None):
```

> DeepLabv3p model training API. The function has a built-in `polynomial` learning rate attenuation policy and a `momentum` optimizer.

> **Parameters**
> - **num_epochs** (int): Number of training iteration epochs.
- **train_dataset** (paddlex.datasets): Training data reader.
- **train_batch_size** (int): Training data batch size. It is also a validation data batch size. It is 2 by default.
- **eval_dataset** (paddlex.datasets): Evaluation data reader.
- **save_interval_epochs** (int): Model saving interval (unit: number of iteration epochs). It is 1 by default.
- **log_interval_steps** (int): Training log output interval (unit: number of iterations). It is 2 by default.
- **save_dir** (str): Path where models are saved. It is 'output' by default.
- **pretrain_weights** (str): If it is a path, a pre-training model under the path is loaded. If it is a string 'IMAGENET', a model weight pre-trained on ImageNet image data is automatically downloaded. If it is a string 'COCO', a model weight pre-trained on the COCO dataset is automatically downloaded (Note: A COCO pre-training model for Xception41, MobileNetV2_x0.25, MobileNetV2_x0.5, MobileNetV2_x1.5 and MobileNetV2_x2.0 is unavailable temporarily). If it is a string 'CITYSCAPES', a model weight pre-trained on the CITYSCAPES dataset is automatically downloaded (Note: A CITYSCAPES pre-training model for Xception41, MobileNetV2_x0.25, MobileNetV2_x0.5, MobileNetV2_x1.5 and MobileNetV2_x2.0 is unavailable temporarily). If it is none, no pre-training model is used. It is 'IMAGENET' by default.
- **optimizer** (paddle.fluid.optimizer): Optimizer. When this parameter is none, the following default optimizer is used: Use the fluid.optimizer. Momentum optimization and the polynomial learning rate attenuation policy.
- **learning_rate** (float): Initial learning rate of the default optimizer. It is 0.01 by default.
- **lr_decay_power** (float): Learning rate attenuation index of the default optimizer. It is 0.9 by default.
- **use_vdl** (bool): Whether to use VisualDL for visualization. It is false by default.
- **sensitivities_file** (str): If it is a path, sensitivity information under the path is loaded to perform pruning. If it is a string 'DEFAULT', sensitivity information obtained on Cityscapes image data is automatically downloaded to perform pruning. If it is None, no pruning is performed. It is None by default.
- **eval_metric_loss** (float): Tolerable precision loss. It is 0.05 by default.
- **early_stop** (bool): Whether to use a policy for early termination of training. It is false by default.
- **early_stop_patience** (int): When a policy for early termination of training is used, training is terminated if the validation set precision continuously decreases or remains unchanged within early_stop_patience epochs. It is 5 by default.``
- **resume_checkpoint** (str): When training is resumed, specify a model path saved during the last training. If it is None, training is not resumed. It is None by default.



### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, return_details=False):
```

> DeepLabv3p model evaluation API.

> **Parameters**
> - **eval_dataset** (paddlex.datasets): Evaluation data reader.
- **batch_size** (int): Batch size during evaluation. It is 1 by default.
- **epoch_id** (int): Number of training epochs of the current evaluation model.
- **return_details** (bool): Whether to return detailed information. It is false by default.



> **Returned value**
> - **dict**: When return_details` is false, dict is returned.`The following keywords are contained: ' miou', 'category_iou', 'macc', 'category_acc' and 'kappa' which indicate the average IoU， the IoU of each class, the average accuracy rate, the accuracy rate of each class and the kappa coefficient respectively.
- **tuple** (metrics, eval_details): When return_details` is true, the return of dict (eval_details) is added.`The following keywords are contained: ' confusion_matrix' which indicates the evaluation confusion matrix.



### predict

```
predict(self, img_file, transforms=None):
```

> DeepLabv3p Model inference API. Note that the image processing flow during inference can be saved in `DeepLabv3p.test_transforms and `DeepLabv3p.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `predict API for prediction is called, you need to redefine and pass test_transforms to the predict API.````

> **Parameters**
> - **img_file** (str|np.ndarray): Path or numpy array of the predicted image (HWC arrangement, BGR format).
- **transforms** (paddlex.seg.transforms): Data preprocessing operation.



> **Returned value**
> - **dict**: It contains the keywords 'label_map' and 'score_map'. 'label_map' stores an inference result grayscale image. A pixel value indicates the corresponding class. 'score_map' stores a probability of each class. shape = (h, w, num_classes).




### batch_predict

```
batch_predict(self, img_file_list, transforms=None):
```

> DeepLabv3p model batch inference API. Note that the image processing flow during inference can be saved in `DeepLabv3p.test_transforms and `DeepLabv3p.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `batch_predict API for prediction is called, you need to redefine and pass test_transforms to the batch_predict API.````

> **Parameters**
> - **img_file_list** (list|tuple): Images in the list (or tuple) are simultaneously predicted. Elements in the list are predicted image paths or numpy arrays (HWC arrangement, BGR format).
- **transforms** (paddlex.seg.transforms): Data preprocessing operation.



> **Returned value**
> - **dict**: Each element is a list which indicates inference results of each image. The inference results of each image is expressed as a dictionary. It contains the keywords 'label_map' and 'score_map'. 'label_map' stores an inference result grayscale image. A pixel value indicates the corresponding class. 'score_map' stores a probability of each class. shape = (h, w, num_classes).




### overlap_tile_predict

```
overlap_tile_predict(self, img_file, tile_size=[512, 512], pad_size=[64, 64], batch_size=32, transforms=None)
```

> Sliding inference API for the DeepLabv3p model. The overlapping and non-overlapping modes are supported.

> **Non-overlapping sliding window inference: Slide on the input image using a window of fixed size. Infer an image under each window. Splice inference results of each window into inference results of the input image.**The parameter pad_size ** must be set to `[0, 0]` during use`.`**

> **Overlapping sliding window inference: In Unet’s paper, the author proposed an overlap-tile strategy to eliminate a crack feeling at the splice.**In the prediction in each sliding window, a certain area is expanded around the expanded window, such as the blue part of the area in the figure below. Only the middle part of the window is predicted in the splice, for example, the yellow part area in the figure below. The pixels under the expanded area of the window located at the edge of the input image are obtained by mirroring the pixels at the edge.

![](../../../examples/remote_sensing/images/overlap_tile.png)

> Note that the image processing flow during inference can be saved in `DeepLabv3p.test_transforms and `DeepLabv3p.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `overlap_tile_predict API for inference is called, you need to redefine and pass test_transforms to the overlap_tile_predict API.````

> **Parameters**
> - **img_file** (str|np.ndarray): Path or numpy array of the predicted image (HWC arrangement, BGR format).
- **tile_size** (list|tuple): Sliding window size. This area is used to splice inference results. The format is (W, H). It is 512, 512[ by default].
- **pad_size** (list|tuple): Size of the area where the sliding window extends towards its surrounding. The extended area is not used to splice prediction results. The format is (W, H). It is 64, 64[ by default].
- **batch_size** (int): Batch size during the batch inference on the window. It is 32 by default.
- **transforms** (paddlex.seg.transforms): Data preprocessing operation.



> **Returned value**
> - **dict**: It contains the keywords 'label_map' and 'score_map'. 'label_map' stores an inference result grayscale image. A pixel value indicates the corresponding class. 'score_map' stores a probability of each class. shape = (h, w, num_classes).



## paddlex.seg.UNet

```python
paddlex. seg.UNet(num_classes=2, upsample_mode='bilinear', use_bce_loss=False, use_dice_loss=False, class_weight=None, ignore_index=255, input_channel=3)
```

> Build a UNet segmenter.

> **Parameters**

> > - **num_classes** (int): Number of classes.
- **upsample_mode** (str): Upsampling mode used during the UNet decoding. When the value is 'bilinear', a bilinear difference value is used to perform upsampling. When other options are input, a deconvolution is used to perform upsasmpling. It is 'bilinear' by default.
- **use_bce_loss** (bool): Whether to use bce loss as a network loss function. The bce loss function can be used for two kinds of segmentation only and may be used with dice loss. It is false by default.
- **use_dice_loss** (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。Whether to use dice loss as a network loss function. The dice loss function can be used for two kinds of segmentation only and may be used with bce loss. When both use_bce_loss and use_dice_loss are false, the cross entropy loss function is used. It is false by default.
- **class_weight** (list/str): Weight of various losses of the cross entropy loss function. When class_weight` is a list, the length shall be `num_classes`.`When class_weight` is str, weight.`lower() shall be 'dynamic'. At this moment, the corresponding weight is automatically calculated according to the proportion of all classes of pixels in each round. The weight of each class is as follows: Proportion of each class * num_classes. When class_weight is the default none, the weight of each class is 1, i.e. the usually used cross entropy loss function.
- **ignore_index** (int): Value ignored on a label. A pixel of which the label is ignore_index does not participate in the calculation of the loss function. It is 255 by default.``
- **input_channel** (int): Number of input image channels. It is 3 by default.



> - The description of the train API for training is the same as the train API of the DeepLabv3p model[](#train)
- The description of the evaluate API for evaluation is the same as the evaluate API of the DeepLabv3p model[](#evaluate)
- The description of the predict API for inference is the same as the predict API of the DeepLabv3p model[](#predict)
- The description of the batch_predict API for batch prediction is the same as the predict API of the DeepLabv3p model[](#batch-predict)
- The overlap_tile_predict API for sliding window prediction is the same as the poverlap_tile_predict API of the DeepLabv3p model[](#overlap-tile-predict)


## paddlex.seg.HRNet

```python
paddlex. seg.HRNet(num_classes=2, width=18, use_bce_loss=False, use_dice_loss=False, class_weight=None, ignore_index=255, input_channel=3)
```

> Build an HRNet segmenter.

> **Parameters**

> > - **num_classes** (int): Number of classes.
- **width** (int|str): Number of channels in the characteristic layer in a high-resolution branch. It is 18 by default. The optional values are 18, 30, 32, 40, 44, 48, 60, 64, '18_small_v1'. ' 18_small_v1' is the lightweight version of 18.[]
- **use_bce_loss** (bool): Whether to use bce loss as a network loss function. The bce loss function can be used for two kinds of segmentation only and may be used with dice loss. It is false by default.
- **use_dice_loss** (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。Whether to use dice loss as a network loss function. The dice loss function can be used for two kinds of segmentation only and may be used with bce loss. When both use_bce_loss and use_dice_loss are false, the cross entropy loss function is used. It is false by default.
- **class_weight** (list|str): Weight of various losses of the cross entropy loss function. When class_weight` is a list, the length shall be `num_classes`.`When class_weight` is str, weight.`lower() shall be 'dynamic'. At this moment, the corresponding weight is automatically calculated according to the proportion of all classes of pixels in each round. The weight of each class is as follows: Proportion of each class * num_classes. When class_weight is the default none, the weight of each class is 1, i.e. the usually used cross entropy loss function.
- **ignore_index** (int): Value ignored on a label. A pixel of which the label is ignore_index does not participate in the calculation of the loss function. It is 255 by default.``
- **input_channel** (int): Number of input image channels. It is 3 by default.



> - The description of the train API for training is the same as the train API of the DeepLabv3p model[](#train)
- The description of the evaluate API for evaluation is the same as the evaluate API of the DeepLabv3p model[](#evaluate)
- The description of the predict API for inference is the same as the predict API of the DeepLabv3p model[](#predict)
- The description of the batch_predict API for batch prediction is the same as the predict API of the DeepLabv3p model[](#batch-predict)
- The overlap_tile_predict API for sliding window inference is the same as the poverlap_tile_predict API of the DeepLabv3p model[](#overlap-tile-predict)


## paddlex.seg.FastSCNN

```python
paddlex. seg.FastSCNN(num_classes=2, use_bce_loss=False, use_dice_loss=False, class_weight=None, ignore_index=255, multi_loss_weight=[1.0], input_channel=3)
```

> Build a FastSCNN segmenter.

> **Parameters**

> > - **num_classes** (int): Number of classes.
- **use_bce_loss** (bool): Whether to use bce loss as a network loss function. The bce loss function can be used for two kinds of segmentation only and may be used with dice loss. It is false by default.
- **use_dice_loss** (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。Whether to use dice loss as a network loss function. The dice loss function can be used for two kinds of segmentation only and may be used with bce loss. When both use_bce_loss and use_dice_loss are false, the cross entropy loss function is used. It is false by default.
- **class_weight** (list/str): Weight of various losses of the cross entropy loss function. When class_weight` is a list, the length shall be `num_classes`.`When class_weight` is str, weight.`lower() shall be 'dynamic'. At this moment, the corresponding weight is automatically calculated according to the proportion of all classes of pixels in each round. The weight of each class is as follows: Proportion of each class * num_classes. When class_weight is the default none, the weight of each class is 1, i.e. the usually used cross entropy loss function.
- **ignore_index** (int): Value ignored on a label. A pixel of which the label is ignore_index does not participate in the calculation of the loss function. It is 255 by default.``
- **multi_loss_weight** (list): Loss weight on multiple branches. The default is to calculate a loss on one branch, i .[e. the default is ]1.0 .[A loss on two or three branches can also be calculated and the weight is arranged in a sequence of ]fusion_branch_weight, higher_branch_weight, lower_branch_weight. fusion_branch_weight is the loss weight on the branch after the spatial detail branch and the global context branch are blended. higher_branch_weight is the loss weight on the spatial detail branch. lower_branch_weight is the loss weight on the global context branch. If higher_branch_weight and lower_branch_weight are not set, a loss on these two branches will not be calculated.
- **input_channel** (int): Number of input image channels. It is 3 by default.



> - The description of the train API for training is the same as the train API of the DeepLabv3p model[](#train)
- The description of the evaluate API for evaluation is the same as the evaluate API of the DeepLabv3p model[](#evaluate)
- The description of the predict API for inference is the same as the predict API of the DeepLabv3p model[](#predict)
- The description of the batch_predict API for batch prediction is the same as the predict API of the DeepLabv3p model[](#batch-predict)
- The overlap_tile_predict API for sliding window inference is the same as the poverlap_tile_predict API of the DeepLabv3p model[](#overlap-tile-predict)

