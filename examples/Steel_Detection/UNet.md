# 基于UNet的精度优化

本小节侧重展示在模型迭代过程中优化精度的思路，在本案例中，有些优化策略获得了精度收益，而有些没有。在其他质检场景中，可根据实际情况尝试这些优化策略。

## (1) 基线模型选择

| arch | backbone |epoch | resolution |  batch size | learning rate |  miou |
| -- | -- | -- | -- | -- | -- | -- |
| DeepLabV3 | ResNet50_vd | 10 | 400x64 | 16 | 0.01 | 48.6%
| HRNET | | 10 | 400x64 | 16 | 0.01 | 43.8%
| UNET | | 10 | 400x64  | 16 | 0.01 | 46.2%
| FastSCNN | | 10 | 400x64 | 16 | 0.01 | 38.9%


## (2) 数据增强选择

从上表可知，不同结构的网络模型，在相同的训练超参数下，表现不同，可知层次较多的模型，精度会比较高，不过层次较深的模型需要的迭代次数也更多，训练时间也更长，仅仅10个epoch是是远远不够的。为了加快训练速度，这里选择了一个结构较为简单的模型UNET作为实验对象，其他模型使用以下方法也是同样有效的。

下面可以加入一些数据增强策略来进一步提升模型的精度。（1）里面只选用了简单的[RandomHorizontalFlip](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomhorizontalflip)数据增强方法，因为数据增强会使样本对样化，所以首先将epoch次数增加到100，可以看到UNET的miou由46.2%->52.2。

| arch  | epoch | resolution |  batch size | learning rate | Augment|  miou |
| -- | -- | -- | -- | -- | -- | -- |
| UNET  | 100 | 400x64 | 16 | 0.01 | RandomHorizontalFlip | 52.2%
| UNET  | 100 | 400x64 | 16 | 0.01 | RandomHorizontalFlip RandomDistort RandomCrop | 53.4%


在上面的基础上，选择加入一些数据增强策略来进一步提升模型的精度。本案例选择同时使用[RandomHorizontalFlip](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomhorizontalflip)、[RandomDistort](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomdistort)、[RandomCrop](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomcrop)这三种数据增强方法，重新训练后的模型在验证集上的miou为53.4%。


## (3) 分类损失函数选择

我们观察样本与标签可知，钢板大部分是没有缺陷的，缺陷仅仅占全部样本很小的一部分，这样就导致了不同标签的样本分布不均匀，这也是影响模型精度的主要原因，我们可以通过选择不同的损失函数或者组合不同的损失函数来优化这个问题，提升模型精度。在paddlex中，已经为用户提供了这个接口，在创建模型是，将use_mixed_loss设置为True，表示使用多种损失函数，实例代码如下:

```
model = pdx.seg.UNet(num_classes=num_classes, use_mixed_loss=True)
```
在Unet中设置use_mixed_loss参数为True，选择的是[CrossEntropyLoss](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.2/docs/module/loss/CrossEntropyLoss_cn.md)和[LovaszSoftmaxLoss](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.2/docs/module/loss/lovasz_loss_cn.md)两种损失函数，他们的权重分别是0.8和0.2。通过这个两种损失函数的组合可缓解样本分布不均衡带来的问题。

| arch  | epoch | resolution |  batch size | learning rate | loss | Augment|  miou |
| -- | -- | -- | -- | -- | -- | -- | -- |
| UNET  | 100 | 400x64 | 16 | 0.01 | CrossEntropyLoss LovaszSoftmaxLoss  | RandomHorizontalFlip RandomDistort RandomCrop | 56.4%

下面还可以进一步通过设置损失函数还提升模型精度，上面通过使用use_mixed_loss=True设置paddlex预先设定好的损失函数，我们也可以自定义损失函数的组合与权重。设置

```
use_mixed_loss=[["CrossEntropyLoss", 0.8], ["DiceLoss", 0.2]]
```

这样就可以自己组合合适的损失函数。这里我们将LovaszSoftmaxLoss调整为[DiceLoss](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.2/docs/module/loss/DiceLoss_cn.md)重新训练，得到以下结果。


| arch  | epoch | resolution |  batch size | learning rate | loss | Augment|  miou |
| -- | -- | -- | -- | -- | -- | -- | -- |
| UNET  | 100 | 400x64 | 16 | 0.01 | CrossEntropyLoss LovaszSoftmaxLoss  | RandomHorizontalFlip RandomDistort RandomCrop | 56.4%
| UNET  | 100 | 400x64 | 16 | 0.01 | CrossEntropyLoss DiceLoss  | RandomHorizontalFlip RandomDistort RandomCrop | 56.7%

可以看到模型有0.3%的提升。

## (4)加入无缺陷的训练样本
将类别都是0（无缺陷）的样本也加入到训练集中，保持验证集不变，重新训练模型，性能提升如下图。

| arch  | epoch | resolution |  batch size | learning rate | loss | Augment|  miou |
| -- | -- | -- | -- | -- | -- | -- | -- |
| UNET  | 100 | 400x64 | 16 | 0.01 | CrossEntropyLoss LovaszSoftmaxLoss  | RandomHorizontalFlip RandomDistort RandomCrop | 56.67%
| UNET  | 100 | 400x64 | 16 | 0.01 | CrossEntropyLoss DiceLoss  | RandomHorizontalFlip RandomDistort RandomCrop | 57.52%
