# 检测模型裁剪

---
本文档训练代码可直接在PaddleX的Repo中下载，[代码tutorials/compress/detection](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/compress/detection)  
本文档按如下方式对模型进行了裁剪
> 第一步：在训练数据集上训练YOLOv3
> 第二步：在验证数据集上计算模型中各个参数的敏感度信息  
> 第三步：根据第二步计算的敏感度，设定`eval_metric_loss`，对模型裁剪后重新在训练数据集上训练

## 步骤一 训练YOLOv3
> 模型训练使用文档可以直接参考[检测模型训练](../train/detection.md)，本文档在该代码基础上添加了部分参数选项，用户可直接下载模型训练代码[tutorials/compress/detection/yolov3_mobilnet.py](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/compress/detection/yolov3_mobilenet.py)  
> 使用如下命令开始模型训练
```
python yolov3_mobilenet.py
```

## 步骤二 计算参数敏感度
> 参数敏感度的计算可以直接使用PaddleX提供的API`paddlex.slim.cal_params_sensitivities`，使用代码如下, 敏感度信息文件会保存至`save_file`

```
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

model = pdx.load_model(model_dir)

# 定义验证所用的数据集
eval_dataset = pdx.datasets.ImageNet(
    data_dir=dataset,
    file_list=os.path.join(dataset, 'val_list.txt'),
    label_list=os.path.join(dataset, 'labels.txt'),
    transforms=model.eval_transforms)

pdx.slim.cal_params_sensitivities(model,
                                save_file,
                                eval_dataset,
                                batch_size=8)
```
> 本步骤代码已整理至[tutorials/compress/detection/cal_sensitivities_file.py](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/compress/detection/cal_sensitivities_file.py)，用户可直接下载使用  
> 使用如下命令开始计算敏感度
```
python cal_sensitivities_file.py --model_dir output/yolov3_mobile/best_model --dataset insect_det --save_file sensitivities.data
```

## 步骤三 开始裁剪训练
> 本步骤代码与步骤一使用同一份代码文件，使用如下命令开始裁剪训练
```
python yolov3_mobilenet.py --model_dir output/yolov3_mobile/best_model --sensitivities_file sensitivities.data --eval_metric_loss 0.10
```

## 实验效果
本教程的实验效果可以查阅[模型压缩文档](../../slim/prune.md)
