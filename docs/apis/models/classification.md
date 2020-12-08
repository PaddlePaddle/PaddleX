# Image Classification

## paddlex.cls. ResNet50

```python
paddlex.cls. ResNet50(num_classes=1000)
```

> Build a ResNet50 classifier and implement its training, evaluation and prediction.

**Parameters**

> - **num_classes** (int): Number of classes. It is 1000 by default.


### train

```python
train(self, num_epochs, train_dataset, train_batch_size=64, eval_dataset=None, save_interval_epochs=1, log_interval_steps=2, save_dir='output', pretrain_weights='IMAGENET', optimizer=None, learning_rate=0.025, warmup_steps=0, warmup_start_lr=0.0, lr_decay_epochs=[30, 60, 90], lr_decay_gamma=0.1, use_vdl=False, sensitivities_file=None, eval_metric_loss=0.05, early_stop=False, early_stop_patience=5, resume_checkpoint=None)
```
> **Parameters**
> > -**num_epochs** (int): Number of training iteration epochs.
> > -**train_dataset** (paddlex.datasets): Training data reader.
> > -**train_batch_size** (int): Training data batch size. It is also a validation data batch size. It is 64 by default.
> > -**eval_dataset** (paddlex.datasets): Validation data reader.
> > -**save_interval_epochs** (int): Model saving interval (unit: number of iteration epochs). It is 1 by default.
> > -**log_interval_steps** (int): Training log output interval (unit: number of iteration steps). It is 2 by default.
> > -**save_dir** (str): Path where models are saved.
> > -**pretrain_weights** (str): If it is a path, a pre-training model under the path is loaded. If it is a string 'IMAGENET', a model weight pre-trained on ImageNet image data is automatically downloaded. If it is None, no pre-training model is used. It is 'IMAGENET' by default.
> > - **optimizer** (paddle.fluid.optimizer): Optimizer. When this parameter is None, a default optimizer is used: fluid.layers.piecewise_decay attenuation policy, fluid.optimizer. Momentum optimization method.
> > -**learning_rate** (float): Initial learning rate of the default optimizer. It is 0.025 by default.
> > - **warmup_steps** (int): Number of warmup steps of the default optimizer. The learning rate will be within a set number of steps and linearly increase to a set learning_rate from warmup_start_lr. It is 0 by default.
> > - **warmup_start_lr**(float): Warmup starting learning rate of the default optimizer. It is 0.0 by default.
> > - **lr_decay_epochs** (list): Number of learning rate attenuation epochs of the default optimizer. It is [30, 60, 90] by default.
> > - **lr_decay_gamma** (float): Attenuation rate of learning rate of the default optimizer. It is 0.1 by default.
> > - **use_vdl** (bool): Whether to use VisualDL for visualization. It is false by default.
> > - **sensitivities_file** (str): If it is a path, sensitivity information under the path is loaded to perform pruning. If it is a string 'DEFAULT', sensitivity information obtained from ImageNet image data is automatically downloaded to perform pruning. If it is None, no pruning is performed. It is None by default.
> > - **eval_metric_loss** (float): Tolerable precision loss. It is 0.05 by default.
> > - **early_stop** (bool): Whether to use a policy for early termination of training. It is false by default.
> > - **early_stop_patience** (int): When a policy for early termination of training is used, training is terminated if the validation set precision continuously decreases or remains unchanged within `early_stop_patience` epochs. It is 5 by default.
> > - **resume_checkpoint** (str): When training is resumed, specify a model path saved during the last training. If it is None, training is not resumed. It is None by default.



### evaluate

```python
evaluate(self, eval_dataset, batch_size=1, epoch_id=None, return_details=False)
```
> **Parameters**
>
> > - **eval_dataset** (paddlex.datasets): Validation data reader.
> > -  **batch_size** (int): Validation data batch size. It is 1 by default.
> > - **epoch_id** (int): Number of training epochs of the current evaluation model.
> > - **return_details** (bool): Whether to return detailed information. It is false by default.
>
>**Returned value**
>
> > -**dict**: When return_details is false, dict is returned, containing keywords: ' acc1' and 'acc5' which indicate the accuracy of the maximum value and the accuracy of top 5 maximum values respectively.
> > - **tuple** (metrics, eval_details): When 'return_details` is true, the return of dict is increased，containing keywords: '`true_labels' and 'pred_scores' which indicate the true class ID and the prediction score of each class respectively.



### predict

```python
predict(self, img_file, transforms=None, topk=1)
```

> Classification model prediction API. Note that the image processing flow during prediction can be saved in `ResNet50.test_transforms' and `ResNet50.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `predict' API for prediction is called, you need to redefine and pass test_transforms to the predict API.

> **Parameters**
> 
> > -**img_file** (str|np.ndarray): Path or numpy array of the predicted image (HWC arrangement, BGR format).
> > -**transforms** (paddlex.cls.transforms): Data preprocessing operation.
> > -**topk** (int): Top k maximum values during prediction.

> **Returned value**
> 
> > -**list**: All elements are dictionaries. Dictionary keywords include 'category_id', 'category' and 'score' 
> >      which correspond to the prediction class ID, prediction class tag and prediction score respectively.



### batch_predict

```python
batch_predict(self, img_file_list, transforms=None, topk=1)
```

> Classification model batch prediction API. Note that the image processing flow during prediction can be saved in `ResNet50.test_transforms' and `ResNet50.eval_transforms` during model saving only when eval_dataset is defined during training. If eval_dataset is not defined during training, when the `batch_predict' API for prediction is called, you need to redefine and pass test_transforms to the batch_predict API.

> **Parameters**
> 
> > -  **img_file_list** (list|tuple): Images in the list (or tuple) are simultaneously predicted. Elements in the list may be image paths or numpy arrays (HWC arrangement, BGR format).
> > - **transforms** (paddlex.cls.transforms): Data preprocessing operation.
> > - **topk** (int): Top k maximum values during prediction.


> **Returned value**
> 
> > - **list**: Each element is a list which indicates prediction results of each image. All elements in the prediction list of images are dictionaries. Dictionary keywords include 'category_id', 'category' and 'score' which correspond to the prediction class ID, prediction class tag and prediction score respectively.


## Other classification models

PaddleX provides a total of 22 classification models. All classification models provide the same `train`, `evaluate` and `predict` APIs as 'ResNet50'. For model effects, refer to the [model library](https://paddlex.readthedocs.io/zh_CN/latest/appendix/model_zoo.html).

| Model | API |
| :---------------- | :---------------------- |
| :---------------- | :---------------------- |
| ResNet18          | paddlex.cls.ResNet18(num_classes=1000) |
| ResNet34          | paddlex.cls.ResNet34(num_classes=1000) |
| ResNet50          | paddlex.cls.ResNet50(num_classes=1000) |
| ResNet50_vd       | paddlex.cls.ResNet50_vd(num_classes=1000) |
| ResNet50_vd_ssld    | paddlex.cls.ResNet50_vd_ssld(num_classes=1000) |
| ResNet101          | paddlex.cls.ResNet101(num_classes=1000) |
| ResNet101_vd        | paddlex.cls.ResNet101_vd(num_classes=1000) |
| ResNet101_vd_ssld      | paddlex.cls.ResNet101_vd_ssld(num_classes=1000) |
| DarkNet53      | paddlex.cls.DarkNet53(num_classes=1000) |
| MoibileNetV1         | paddlex.cls.MobileNetV1(num_classes=1000) |
| MobileNetV2       | paddlex.cls.MobileNetV2(num_classes=1000) |
| MobileNetV3_small       | paddlex.cls.MobileNetV3_small(num_classes=1000) |
| MobileNetV3_small_ssld  | paddlex.cls.MobileNetV3_small_ssld(num_classes=1000) |
| MobileNetV3_large   | paddlex.cls.MobileNetV3_large(num_classes=1000) |
| MobileNetV3_large_ssld | paddlex.cls.MobileNetV3_large_ssld(num_classes=1000) |
| Xception65     | paddlex.cls.Xception65(num_classes=1000) |
| Xception71     | paddlex.cls.Xception71(num_classes=1000) |
| ShuffleNetV2     | paddlex.cls.ShuffleNetV2(num_classes=1000) |
| DenseNet121      | paddlex.cls.DenseNet121(num_classes=1000) |
| DenseNet161       | paddlex.cls.DenseNet161(num_classes=1000) |
| DenseNet201       | paddlex.cls.DenseNet201(num_classes=1000) |
| HRNet_W18       | paddlex.cls.HRNet_W18(num_classes=1000) |
| AlexNet         | paddlex.cls.AlexNet(num_classes=1000) |
