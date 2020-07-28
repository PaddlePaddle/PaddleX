# 使用教程——模型压缩
本目录下整理了使用PaddleX进行模型剪裁训练的代码，代码均会自动下载数据，并使用单张GPU卡进行训练。

PaddleX提供了两种剪裁训练方式，  
1. 用户自行计算剪裁配置(推荐)，整体流程为
> 1.使用数据训练原始模型；
> 2.使用第1步训练好的模型，在验证集上计算各个模型参数的敏感度，并将敏感信息保存至本地文件
> 3.再次使用数据训练原始模型，在训练时调用`train`接口时，传入第2步计算得到的参数敏感信息文件，
> 4.模型在训练过程中，会根据传入的参数敏感信息文件，对模型结构剪裁后，继续迭代训练
>
2. 使用PaddleX预先计算好的参数敏感度信息文件，整体流程为
> 1. 在训练调用`train`接口时，将`sensetivities_file`参数设为`DEFAULT`字符串
> 2. 在训练过程中，会自动下载PaddleX预先计算好的模型参数敏感度信息，并对模型结构剪裁，继而迭代训练

上述两种方式，第1种方法相对比第2种方法少了两步（即用户训练原始模型+自行计算参数敏感度信息)，实验验证第1种方法的精度会更高，剪裁的模型效果更好，因此在时间和计算成本允许的前提下，更推荐使用第1种方法。


## 开始剪裁训练

* 第1种方法，用户自行计算剪裁配置

```
# 训练模型
python classification/mobilenetv2.py
# 计算模型参数敏感度
python classification/cal_sensitivities_file.py --model_dir=output/mobilenetv2/best_model --save_file=./sensitivities.data
# 剪裁训练
python classification/mobilenetv2.py  --model_dir=output/mobilenetv2/best_model --sensitivities_file=./sensitivities.data --eval_metric_loss=0.05
```

* 第2种方法，使用PaddleX预先计算好的参数敏感度文件

```
# 自动下载PaddleX预先在ImageNet上计算好的参数敏感度信息文件
python classification/mobilenetv2.py --sensitivities_file=DEFAULT --eval_metric_loss=0.05
```
