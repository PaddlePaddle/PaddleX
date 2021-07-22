# 模型裁剪

模型裁剪可以更好地满足在端侧、移动端上部署场景下的性能需求，可以有效得降低模型的体积，以及计算量，加速预测性能。PaddleX集成了PaddleSlim的基于敏感度的通道裁剪算法，用户可以在PaddleX的训练代码里轻松使用起来。

在本文档中展示了分类模型的裁剪过程，文档中代码以及更多其它模型的的裁剪代码可在Github中的[tutorials/slim/prune](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/slim/prune)目录获取。


## 使用方法

模型裁剪相对比我们普通训练一个模型，步骤会多出两步

- 1.采用正常的方式训练一个模型  
- 2.对模型的参数进行敏感度分析
- 3.根据第2步得到的敏感度信息，对模型进行裁剪，并以第1步训练好的模型作为预训练权重，继续进行训练

具体我们以图像分类模型MobileNetV2为例，本示例中所有代码均可在Github的[tutorials/slim/prune/image_classification]中获得。

### 第一步 正常训练模型

此步骤中采用正常的代码进行模型训练，在获取[本示例代码](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/slim/prune/image_classification)后，直接执行如下命令即可
```
python mobilenetv2_train.py
```
在训练完成后，我们以`output/mobilenetv2/best_model`保存的模型，继续接下来的步骤


### 第二步 参数敏感度分析

此步骤中，我们需要加载第一步训练保存的模型，并通过不断地遍历参数，分析各参数裁剪后在验证数据集上的精度损失，以此判断各参数的敏感度。敏感度分析的代码很简单, 用户可直接查看[params_analysis.py](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/slim/prune/image_classification)。在命令行终端执行如下命令开始参数分析。
```
python params_analysis.py
```

在此步骤中，我们会得到保存的`mobilenetv2.sensi.data`文件，这个文件保存了模型中每个参数的敏感度，在后续的裁剪训练中，会根据此文件中保存的信息，对各个参数进行裁剪。同时，我们也可以对这个文件进行可视化分析，判断`eval_metric_loss`的大小设置与模型被裁剪比例的关系。（`eval_metric_loss`的说明见第三步）

模型裁剪比例可视化分析代码见[slim_visualize.py](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/slim/prune/image_classification)，执行如下命令即可
```
python slim_visualize.py
```
可视化结果如下，该图表明，当我们将`eval_metric_loss`设为0.05时，模型将被裁剪掉65%；将`eval_metric_loss`设为0.10，模型将被裁剪掉68.0%。因此在实际使用时，我们可以根据自己的需求，去设置`eval_metric_loss`控制裁剪比例。

### 第三步 模型裁剪训练

在前两步，我们得到了正常训练保存的模型`output/mobilenetv2/best_model`和基于该保存模型得到的参数敏感度信息文件`mobilenetv2.sensi.data`，接下来则是进行模型裁剪训练。  
模型裁剪训练的代码第第一步基本一致，唯一区别在最后的`train`函数中，我们修改了`pretrain_weights`，`save_dir`，`sensitivities_file`和`eval_metric_loss`四个参数，如下所示
```
model.train(
	num_epoch=10,
	train_dataset=train_dataset,
	train_batch_size=32,
	eval_dataset=eval_dataset,
	lr_decay_epochs=[4,6,8],
	learning_rate=0.025,
	pretrain_weights='output/mobilenetv2/best_model',
	save_dir='output/mobilenetv2_prune',
	sensitivities_file='./mobilenetv2.sensi.data',
	eval_metric_loss=0.05,
	use_vdl=True)
```
具体代码见[tutorials/slim/prune/image_classification/mobilenetv2_prune_train.py](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/slim/prune/image_classification)，执行如下命令即可
```
python mobilenetv2_prune_train.py
```
其中修改的4个参数函数如下
- pretrain_weights: 预训练权重，在裁剪训练中，将其指定为第一步正常训练得到的模型路径
- save_dir: 裁剪训练过程中，模型保存的新路径
- sensitivities_file: 第二步中分析得到的各参数敏感度信息文件
- eval_metric_loss: 可用于控制模型最终被裁剪的比例，见第二步中的可视化说明

## 裁剪效果

在本示例的数据集上，经过裁剪训练后，模型的效果对比如下，其中预测速度不包括图像的预处理和结果的后处理。  
从表中可以看到，对于本示例中的简单数据集，模型裁剪掉68%后，模型准确度没有降低，在CPU的单张图片预测用时减少了37%

| 模型 | 参数大小 | CPU预测速度(MKLDNN关闭) | 准确率 |
| :--- | :----- | :-------------------- | :--- |
| output/mobilenetv2/best_model | 8.7M | 0.057s | 0.92 |
| output/mobilenetv2_prune/best_model | 2.8M | 0.036s | 0.99 |
