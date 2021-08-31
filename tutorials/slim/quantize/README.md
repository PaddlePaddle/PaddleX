# 模型量化


这里**以图像分类为例**展示模型量化的使用过程，示例代码位于[image_classification/mobilenetv2_train.py](image_classification/mobilenetv2_train.py)和[image_classification/mobilenetv2_qat.py](image_classification/mobilenetv2_qat.py)，目标检测、实例分割和语义分割的量化使用均与图像分类一致，皆可参考以下步骤完成模型的在线量化。


## 第一步 正常训练图像分类模型

```
python mobilenetv2_train.py
```

在此步骤中，训练的模型会保存在`output/mobilenet_v2`目录下


## 第二步 模型在线量化

```
python mobilenetv2_qat.py
```

`mobilenetv2_qat.py`中主要执行了以下API：

step 1: 加载之前训练好的模型


```python
model = pdx.load_model('output/mobilenet_v2/best_model')
```

step 2: 完成在线量化

```python
model.quant_aware_train(
    num_epochs=5,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    learning_rate=0.000025,
    save_dir='output/mobilenet_v2/quant',
    use_vdl=True)
```

量化训练后的模型保存在`output/mobilenet_v2/quant`。

**注意：** 重新训练时需将`pretrain_weights`设置为`None`，否则模型会加载`pretrain_weights`指定的预训练模型参数。
