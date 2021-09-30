# 实例分割模型量化

这里展示了实例分割模型量化的过程，示例代码位于[mask_rcnn_train.py](./mask_rcnn_train.py)和[mask_rcnn_qat.py](./mask_rcnn_qat.py)。


## 第一步 正常训练图像分类模型

```
python mask_rcnn_train.py
```

在此步骤中，训练的模型会保存在`output/mask_rcnn_r50_fpn`目录下


## 第二步 模型在线量化

```
python mask_rcnn_qat.py
```

`mask_rcnn_qat.py`中主要执行了以下API：

step 1: 加载之前训练好的模型


```python
model = pdx.load_model('output/mask_rcnn_r50_fpn/best_model')
```

step 2: 完成在线量化

```python
model.quant_aware_train(
    num_epochs=6,
    train_dataset=train_dataset,
    train_batch_size=1,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    save_dir='output/mask_rcnn_r50_fpn/quant',
    use_vdl=True)
```

量化训练后的模型保存在`output/mask_rcnn_r50_fpn/quant`。

**注意：** 重新训练时需将`pretrain_weights`设置为`None`，否则模型会加载`pretrain_weights`指定的预训练模型参数。
