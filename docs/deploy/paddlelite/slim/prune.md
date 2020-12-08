# Model pruning

To meet the requirements of low memory, low bandwidth, low power consumption, low computing resource usage, and low model storage in end-side deployment scenarios, PaddleX is integrated with PaddleSlim to implement the model pruning function, and to further improve the Paddle Lite end-side deployment performance.

## Introduction to the principles

Model pruning is used to reduce the size of the model and reduce the computational complexity of the model by pruning the size of the Kernel output channel in the convolutional layer and the size of its related layer parameters. This can accelerate the prediction speed after model deployment. The principle of related pruning can be found in [PaddleSlim document](https://paddlepaddle.github.io/PaddleSlim/algo/algo.html#id16). **Generally, under the premise of the same model precision, the lower the data complexity, the higher the proportion of models that can be pruned**.

## Pruning method
PaddleX provides two pruning methods:

**1. You can calculate the pruning configuration (recommended). The overall process has three steps:**

* **Step 1**: Use the dataset to train the original model**
* **Step 2**: Use the model trained in Step 1 to calculate the sensitivity of each parameter in the model on the validation dataset, and store the sensitivity information to a local file**
* **Step 3**: Use the dataset to `train` the pruning model (the difference from Step 1 is that the sensitive information file calculated in Step 2 needs to be passed to the `sensitivities_file` parameter of the interface in the train interface)

> In the above three steps, the model needs to be trained twice. The first model training corresponds to Step 1, and the second model training corresponds to Step 3. It is the pruned model in Step 3; therefore, the training speed is faster than Step 1. 
> In Step 2, some of the tailoring parameters in the model are traversed, to calculate the impact of each parameter pruning on the model of the validation set. **Therefore, the validation set is evaluated for multiple times**.

**2. Use PaddleX's built-in pruning scheme**
> The built-in model pruning scheme of PaddleX is **based on the parameter sensitivity information computed on the public dataset** commonly used in each task. Because the feature distribution of different datasets vary greatly, the model **accuracy obtained by this scheme is generally lower** than the first scheme (**the greater the difference between the user-defined dataset and the standard dataset feature distribution, the lower the accuracy of the training model). It can be used as a reference only if a user wants to save time. Only one step is required.

> **One step**: Use the dataset to train the pruning model. When the `train` interface is called for training, set the `sensitivities_file` parameter in the interface to the `DEFAULT`.

> Note: The built-in pruning scheme of each model is based on the public datasets respectively: image classification-ImageNet dataset, object detection-PascalVOC dataset, semantic segmentation-CityScape dataset.

## Pruning experiment
Based on the above two schemes, the experiment is conducted on PaddleX by using example data. The experimental indicators on Tesla P40 are as follows:

### Image classification
Experimental background: Use the MobileNetV2 model. The data set is the sample data of vegetable classification. For the pruning training code, see [tutorials/compress/classification](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/compress/classification).

| Model | Pruning | Model Size | Top1 Accuracy Rate (%) | GPU Prediction Speed | CPU Prediction Speed |
| :-----| :--------| :-------- | :---------- |:---------- |:----------|
| MobileNetV2 | No pruning (original model) | 13.0M | 97.50 | 6.47ms | 47.44ms |
| MobileNetV2 | Scheme 1 (eval_metric_loss=0.10) | 2.1M | 99.58 | 5.03ms | 20.22ms |
| MobileNetV2 | Scheme 2 (eval_metric_loss=0.10) | 6.0M | 99.58 | 5.42ms | 29.06ms |

### Object detection
Experimental background: Use the YOLOv3-MobileNetV1 model. The dataset is insect detection example data. For the pruning training codes, see [tutorials/compress/detection]. (https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/compress/detection)

| Model | Pruning | Model size | MAP(%) | GPU Prediction Speed | CPU Prediction Speed |
| :-----| :--------| :-------- | :---------- |:---------- | :---------|
| YOLOv3-MobileNetV1 | No pruning (original model) | 139M | 67.57 | 14.88ms | 976.42ms |
| YOLOv3-MobileNetV1 | Scheme 1 (eval_metric_loss=0.10) | 34M | 75.49 | 10.60ms | 558.49ms |
| YOLOv3-MobileNetV1 | Scheme 2 (eval_metric_loss=0.05) | 29M | 50.27 | 9.43ms | 360.46ms |

### Semantic segmentation
Experimental background: Use the UNet model. The dataset is the example data of optic disc segmentation. For the pruning training code, see [tutorials/compress/segmentation](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/compress/segmentation).

| Model | Pruning | Model Size | mIoU(%) | GPU Prediction Speed | CPU Prediction Speed |
| :-----| :--------| :-------- | :---------- |:---------- | :---------|
| UNet | No pruning (original model) | 77M | 91.22 | 33.28ms | 9523.55ms |
| UNet | Scheme 1 (eval_metric_loss=0.10) | 26M | 90.37 | 21.04ms | 3936.20ms |
| UNet | Scheme 2 (eval_metric_loss=0.10) | 23M | 91.21 | 18.61ms | 3447.75ms |
