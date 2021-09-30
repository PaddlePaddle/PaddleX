# 目标检测模型量化

这里展示了目标检测模型量化的过程，示例代码位于[yolov3_train.py](./yolov3_train.py)和[yolov3_qat.py](./yolov3_qat.py)。


## 第一步 正常训练图像分类模型

```
python yolov3_train.py
```

在此步骤中，训练的模型会保存在`output/yolov3_darknet53`目录下


## 第二步 模型在线量化

```
python yolov3_qat.py
```

`yolov3_qat.py`中主要执行了以下API：

step 1: 加载之前训练好的模型


```python
model = pdx.load_model('output/yolov3_darknet53/best_model')
```

step 2: 完成在线量化

```python
model.quant_aware_train(
    num_epochs=50,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.0001 / 8,
    warmup_steps=100,
    warmup_start_lr=0.0,
    save_interval_epochs=1,
    lr_decay_epochs=[30, 45],
    save_dir='output/yolov3_darknet53/quant',
    use_vdl=True)
```

量化训练后的模型保存在`output/yolov3_darknet53/quant`。

**注意：** 重新训练时需将`pretrain_weights`设置为`None`，否则模型会加载`pretrain_weights`指定的预训练模型参数。
