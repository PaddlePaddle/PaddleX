# paddlex.load_model模型加载

## 目录

* [加载模型用于训练](#1)
* [加载模型用于评估](#2)
* [加载模型用于剪裁](#3)
* [加载模型用于量化](#4)
* [加载模型用于预测](#5)

## <h2 id="1">加载模型用于训练</h2>

我们以图像分类模型`MobileNetV3_small`为例，假设我们之前训练并保存好了模型（训练代码可参考[示例代码](../../../tutorials/train/image_classification/mobilenetv3_small.py)），在这次训练时想加载之前训好的参数（之前训好的模型假设位于`output/mobilenetv3_small/best_model`），有两种实现方式：

方式一： 使用paddlex.load_model

```python
import paddlex as pdx

model = pdx.load_model("output/mobilenetv3_small/best_model")

model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.01,
    save_dir='output/mobilenetv3_small_new',
    use_vdl=True)
```

方式二： 指定pretrain_weights

```python
import paddlex as pdx

model = pdx.cls.MobileNetV3_small(num_classes=num_classes)

model.train(
    pretrain_weights='output/mobilenetv3_small/best_model/model.pdparams',
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.01,
    save_dir='output/mobilenetv3_small_new',
    use_vdl=True)
```

**注意**：`paddlex.load_model`只加载模型参数但不会恢复优化器设置，如果想要恢复训练，需定义模型之后在调用`train()`时指定[`resume_checkpoint`](./classification.md#train)为`output/mobilenetv3_small/best_model`。

## <h2 id="2">加载模型用于评估</h2>

我们以图像分类模型`MobileNetV3_small`为例，假设我们之前训练并保存好了模型（训练代码可参考[示例代码](../../../tutorials/train/image_classification/mobilenetv3_small.py)），在这次想加载之前训好的参数（之前训好的模型假设位于`output/mobilenetv3_small/best_model`）重新评估模型在验证集上的精度，示例代码如下：

```python
import paddlex as pdx
from paddlex import transforms as T


eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()
])

eval_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/val_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=eval_transforms)


model = pdx.load_model("output/mobilenetv3_small/best_model")

res = model.evaluate(eval_dataset, batch_size=2)
print(res)
```

`evaluate`参数和返回值格式可参考[evaluate](./classification.md#evaluate)

## <h2 id="3">加载模型用于剪裁</h2>

模型剪裁时，先使用`paddlex.load_moel`加载模型，而后使用`analyze_sensitivity`、`prune`和`train`三个API完成剪裁：
```python
model = pdx.load_model('output/mobilenet_v2/best_model')

model.analyze_sensitivity(
    dataset=eval_dataset, save_dir='output/mobilenet_v2/prune')
model.prune(pruned_flops=.2, save_dir=None)

model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.025,
    pretrain_weights=None,
    save_dir='output/mobilenet_v2/prune',
    use_vdl=True)

```
具体的代码请参考[模型剪裁示例代码](../../../tutorials/slim/prune/image_classification/)

## <h2 id="4">加载模型用于量化</h2>

模型量化时，先使用`paddlex.load_moel`加载模型，而后使用`quant_aware_train`完成量化：

```python
model = pdx.load_model('output/mobilenet_v2/best_model')

model.quant_aware_train(
    num_epochs=5,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    learning_rate=0.000025,
    save_dir='output/mobilenet_v2/quant',
    use_vdl=True)
```

具体的代码请参考[模型量化示例代码](../../../tutorials/slim/quantize/)

## <h2 id="5">加载模型用于预测</h2>

请转至文档[加载模型预测](../prediction.md)查看加载模型用于预测的使用方式。
