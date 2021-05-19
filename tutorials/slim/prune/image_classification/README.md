# 图像分类模型剪裁


## 第一步 正常训练图像分类模型

```
python mobilenetv2_train.py
```

在此步骤中，训练的模型会保存在`output/mobilenet_v2`目录下


## 第二步 模型剪裁

```
python mobilenetv2_prune.py
```

`mobilenetv2_prune.py`中主要执行了以下API：

step 1: 分析模型各层参数对于不同的模型FLOPs减小比例下的敏感度

主要由两个API完成:

```
model = pdx.load_model('output/mobilenet_v2/best_model')
model.analyze_sensitivity(
    dataset=eval_dataset, save_dir='output/mobilenet_v2/prune')
```

参数分析完后，`output/mobilenet_v2/prune`目录下会得到`model.sensi.data`文件，此文件保存了不同剪裁比例下各层参数的敏感度信息。

**注意：** 如果之前运行过该步骤，第二次运行时会自动加载已有的`output/mobilenet_v2/prune/model.sensi.data`，不再进行敏感度分析。

step 2: 根据选择的FLOPs减小比例对模型进行剪裁

```
model.prune(pruned_flops=.2, save_dir=None)
```

**注意：** 如果想直接保存剪裁完的模型参数，设置`save_dir`即可。但我们强烈建议对剪裁过的模型重新进行训练，以保证模型精度损失能尽可能少。


step 3: 对剪裁后的模型重新训练

```
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

重新训练后的模型保存在`output/mobilenet_v2/prune`。

**注意：** 重新训练时需将`pretrain_weights`设置为`None`，否则模型会加载`pretrain_weights`指定的预训练模型参数。
