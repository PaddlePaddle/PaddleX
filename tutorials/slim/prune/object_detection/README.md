# 目标检测模型剪裁


## 第一步 正常训练目标检测模型

```
python yolov3_train.py
```

在此步骤中，训练的模型会保存在`output/yolov3_darknet53`目录下


## 第二步 模型剪裁

**注意**：目标检测模型的剪裁依赖PaddleSlim 2.1.0

```
python yolov3_prune.py
```

`yolov3_prune.py`中主要执行了以下API：

step 1: 分析模型各层参数在不同的剪裁比例下的敏感度

主要由两个API完成:

```python
model = pdx.load_model('output/yolov3_darknet53/best_model')
model.analyze_sensitivity(
    dataset=eval_dataset,
    batch_size=1,
    save_dir='output/yolov3_darknet53/prune')
```

参数分析完后，`output/yolov3_darknet53/prune`目录下会得到`model.sensi.data`文件，此文件保存了不同剪裁比例下各层参数的敏感度信息。

**注意：** 如果之前运行过该步骤，第二次运行时会自动加载已有的`output/yolov3_darknet53/prune/model.sensi.data`，不再进行敏感度分析。

step 2: 根据选择的FLOPs减小比例对模型进行剪裁

```python
model.prune(pruned_flops=.2, save_dir=None)
```

**注意：** 如果想直接保存剪裁完的模型参数，设置`save_dir`即可。但我们强烈建议对剪裁过的模型重新进行训练，以保证模型精度损失能尽可能少。


step 3: 对剪裁后的模型重新训练

```python
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.001 / 8,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    lr_decay_epochs=[216, 243],
    save_dir='output/yolov3_darknet53/prune')

```

重新训练后的模型保存在`output/yolov3_darknet53/prune`。

**注意：** 重新训练时需将`pretrain_weights`设置为`None`，否则模型会加载`pretrain_weights`指定的预训练模型参数。
