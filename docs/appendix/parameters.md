# Training parameter adjustment

In all the training interfaces of PaddleX, the built-in parameters are based on the better parameters under the corresponding batch_size of the single GPU card. Users train the model on their own data. When a parameter needs to be adjusted, refer to the following modes (in case of lack of rich parameter adjustment experiences):

## 1. Adjustment of num_epochs
num_epochs: The total number of rounds of training iterations of the model (the model goes through all the samples of the training set once is an epoch). The user can set to a larger value to determine whether the model converges or not based on the performance of the model's iterations on the validation set, and then terminate the training early. In addition, you can also use the `early_stop` strategy in the `train` interface. The model automatically determines whether the model converges and aborts automatically.

## 2. batch_size and learning_rate

> - Batch Size refers to the number of samples used to compute the model forward once (that is, one step) during training.
> - If you are training with multiple cards, the batch_size is divided equally among the cards (so, you need to divide the batch size by the number of cards).
> - Batch Size: It is highly related to the video/memory. The higher the `batch_size`, the more video/memory is consumed.
> - PaddleX configures the default batch size (by default, for single GPU card) in each `train` interface. If the system prompts insufficient GPU memory in the training, you should set BatchSize to a smaller value accordingly.
> - **If the user adjusts the batch size, the user should also adjust other parameters, especially the default learning_rate value in the train interface**. For example, in YOLOv3 model, the default training_batch_size is 8 and `learning_rate` is 0.000125. When you train the model on a card No.2, you can set training_batch_size to 16, and then learning_rate to 0.000125 * 2 = 0.00025` `````


## 3.warmup_steps and warmup_start_lr

In the model training, the pre-training model is usually used. For example, use backbone's pre-training weights on the ImageNet dataset during training in the detection model. However, due to the large difference between own data and the ImageNet dataset in the training, the training may have problems at first for the large step. In this case, you can set learning rate to a smaller value, and then grow slowly to a proper learning rate. `Warmup_steps and `warmup_start_lr` are used for this purpose. When the model starts training, the learning rate starts from warmup_start_lr and grows linearly to the set learning rate after iterations of `warmup_steps` and the batch data.```

> For example, in the train interface of YOLOv3, the default `train_batch_size` is 8, `learning_rate` is 0.000125, `warmup_steps` is 1000, and `warmup_start_lr` is 0.0. With this parameter configuration, after the model starts training, the learning rate grows linearly from 0.0 to a set value 0.000125 after the first 1000 steps (each step uses one batch of data, that is, 8 samples), the learning rate grows linearly from 0.0 to a set 0.000125.

## 4. lr_decay_epochs and lr_decay_gamma

`lr_decay_epochs` is used to allow the learning rate to decay progressively later in the model training; it is typically a list such as 6, 8, and 10, indicating that the learning rate decays once at the 6th epoch, again at the 8th epoch, and again at the 10th epoch.[Each learning rate decays as the previous learning rate * lr_decay_gamma.]

> For example`,` the train interface of YOLOv3 has a default `num_epochs` of 270, `learning_rate` of 0.000125, [lr_decay_epochs] of `213, 240`, and lr_decay_gamma of 0.1. In this parameter configuration, after the model starts training, for the first 213 epochs, the learning rate used for training is 0.000125, for 213-240 epochs, the learning rate in training is 0.000125x0.1=0.0000125, and for more than 240 epochs, the learning rate is 0.000125x0.1x0.1=0.00000125.

## 5. Constraints on parameter setting
Based on these several parameters, it is understood that the change of the learning rate includes WarmUp and Decay.
> - Wamup: with training iterations, the learning rate grows linearly from a low value to a set value. The unit is step. .
> - Decay phase: with training iterations, the learning rate gradually decays, that is, each decay is 0.1 of the previous one. The unit is epoch. .
> - The relationship between step and epoch: 1 epoch is composed of several steps. For example, there are 800 images in the training sample, and the train_batch_size` is 8, and each epoch should use these 800 images to train the model once, and each epoch contains 800//8=100 steps in total.`


In PaddleX, the constraint warmup must end before Decay, so each parameter needs to meet the following conditions.
```
warmup_steps <= lr_decay_epochs[0] * num_steps_each_epoch
```
where `num_steps_each_epoch` is calculated as follows,
```
num_steps_each_eposh = num_samples_in_train_dataset // train_batch_size
```

Therefore, if you are prompted “warmup_steps should be less than” in the start of the training,`. . it means that you need to adjust your parameters according to the above formula: `lr_decay_epochs` or `warmup_steps`.`

## 6. How to use multi-GPU cards for training
Configure environment variables in front of `import paddlex`. Codes are as follows:
```
import os os. environ['CUDA_VISIBLE_DEVICES'] = '0' # Training with GPU card 0 # Note that either paddle or paddlex needs to be imported after setting environment variables. import paddlex as pdx
```

```
import os os. environ['CUDA_VISIBLE_DEVICES'] = '' # Use the CPU for training without using the GPU import paddlex as pdx
```

```
import os os. environ['CUDA_VISIBLE_DEVICES'] = '0,1,3' # Simultaneous training with GPU cards 0, 1, and 3 import paddlex as pdx
```


## Related Model Interface

- Image classification model train interface[](../apis/models/classification.html#train)
- Object detection FasterRCNN train interface[](../apis/models/detection.html#id1)
- Object detection YOLOv3 train interface[](../apis/models/detection.html#train)
- Instance segmentation MaskRCNN train interface[](../apis/models/instance_segmentation.html#train)
- Semantic segmentation train interface[](../apis/models/semantic_segmentation.html#train)
