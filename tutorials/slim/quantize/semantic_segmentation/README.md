# 语义分割模型量化

这里展示了语义分割模型量化的过程，示例代码位于[unet_train.py](./unet_train.py)和[unet_qat.py](./unet_qat.py)。


## 第一步 正常训练图像分类模型

```
python unet_train.py
```

在此步骤中，训练的模型会保存在`output/unet`目录下


## 第二步 模型在线量化

```
python unet_qat.py
```

`unet_qat.py`中主要执行了以下API：

step 1: 加载之前训练好的模型


```python
model = pdx.load_model('output/unet/best_model')
```

step 2: 完成在线量化

```python
model.quant_aware_train(
    num_epochs=5,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.001,
    save_dir='output/unet/quant',
    use_vdl=True)
```

量化训练后的模型保存在`output/unet/quant`。

**注意：** 重新训练时需将`pretrain_weights`设置为`None`，否则模型会加载`pretrain_weights`指定的预训练模型参数。
