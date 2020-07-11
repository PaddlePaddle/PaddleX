# 10分钟快速上手使用

本文档在一个小数据集上展示了如何通过PaddleX进行训练，您可以阅读PaddleX的**使用教程**来了解更多模型任务的训练使用方式。本示例同步在AIStudio上，可直接[在线体验模型训练](https://aistudio.baidu.com/aistudio/projectdetail/439860)

PaddleX中的所有模型训练跟随以下3个步骤，即可快速完成训练代码开发！

| 步骤 |                  |说明             |
| :--- | :--------------- | :-------------- |
| 第1步| <a href=#定义训练验证图像处理流程transforms>定义transforms</a>  | 用于定义模型训练、验证、预测过程中，<br>输入图像的预处理和数据增强操作 |
| 第2步| <a href="#定义dataset加载图像分类数据集">定义datasets</a>  | 用于定义模型要加载的训练、验证数据集 |
| 第3步| <a href="#使用MoibleNetV3_small_ssld模型开始训练">定义模型开始训练</a> | 选择需要的模型，进行训练 |

> **注意**：不同模型的transforms、datasets和训练参数都有较大差异，更多模型训练，可直接根据文档教程获取更多模型的训练代码。[模型训练教程](train/index.html)

PaddleX的其它用法

- <a href="#训练过程使用VisualDL查看训练指标变化">使用VisualDL查看训练过程中的指标变化</a>
- <a href="#加载训练保存的模型预测">加载训练保存的模型进行预测</a>


<a name="安装PaddleX"></a>
**1. 安装PaddleX**  
> 安装相关过程和问题可以参考PaddleX的[安装文档](./install.md)。
```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

<a name="准备蔬菜分类数据集"></a>
**2. 准备蔬菜分类数据集**  
```
wget https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz
tar xzvf vegetables_cls.tar.gz
```

<a name="定义训练验证图像处理流程transforms"></a>
**3. 定义训练/验证图像处理流程transforms**  

由于训练时数据增强操作的加入，因此模型在训练和验证过程中，数据处理流程需要分别进行定义。如下所示，代码在`train_transforms`中加入了[RandomCrop](apis/transforms/cls_transforms.html#RandomCrop)和[RandomHorizontalFlip](apis/transforms/cls_transforms.html#RandomHorizontalFlip)两种数据增强方式, 更多方法可以参考[数据增强文档](apis/transforms/augment.md)。
```
from paddlex.cls import transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])
```

<a name="定义dataset加载图像分类数据集"></a>
**4. 定义`dataset`加载图像分类数据集**  

定义数据集，`pdx.datasets.ImageNet`表示读取ImageNet格式的分类数据集
- [paddlex.datasets.ImageNet接口说明](apis/datasets/classification.md)
- [ImageNet数据格式说明](data/format/classification.md)

```
train_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/train_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/val_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=eval_transforms)
```

<a name="使用MoibleNetV3_small_ssld模型开始训练"></a>
**5. 使用MobileNetV3_small_ssld模型开始训练**  

本文档中使用百度基于蒸馏方法得到的MobileNetV3预训练模型，模型结构与MobileNetV3一致，但精度更高。PaddleX内置了20多种分类模型，查阅[PaddleX模型库](appendix/model_zoo.md)了解更多分类模型。
```
num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_small_ssld(num_classes=num_classes)

model.train(num_epochs=20,
            train_dataset=train_dataset,
            train_batch_size=32,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_dir='output/mobilenetv3_small_ssld',
            use_vdl=True)
```

<a name="训练过程使用VisualDL查看训练指标变化"></a>
**6. 训练过程使用VisualDL查看训练指标变化**  

模型在训练过程中，训练指标和在验证集上的指标，均会以标准输出流形式，输出到命令终端。在用户设定`use_vdl=True`的前提下，也会使用VisualDL格式打点到`save_dir`目录下的`vdl_log`文件夹，用户可都终端通过如下命令启动visualdl，查看可视化的指标变化趋势。
```
visualdl --logdir output/mobilenetv3_small_ssld --port 8001
```
服务启动后，通过浏览器打开https://0.0.0.0:8001或https://localhost:8001即可。

> 如果您使用的是AIStudio平台进行训练，不能通过此方式启动visualdl，请参考AIStudio VisualDL启动教程使用

<a name="加载训练保存的模型预测"></a>
**7. 加载训练保存的模型预测**  

模型在训练过程中，会每间隔一定轮数保存一次模型，在验证集上评估效果最好的一轮会保存在`save_dir`目录下的`best_model`文件夹。通过如下方式可加载模型，进行预测。
- [load_model接口说明](apis/load_model.md)
- [分类模型predict接口说明](apis/models/classification.html#predict)
```
import paddlex as pdx
model = pdx.load_model('output/mobilenetv3_small_ssld/best_model')
result = model.predict('vegetables_cls/bocai/100.jpg')
print("Predict Result: ", result)
```
预测结果输出如下,
```
Predict Result: Predict Result: [{'score': 0.9999393, 'category': 'bocai', 'category_id': 0}]
```

<a name="更多使用教程"></a>
**更多使用教程**
- 1.[目标检测模型训练](tutorials/train/detection.md)
- 2.[语义分割模型训练](tutorials/train/segmentation.md)
- 3.[实例分割模型训练](tutorials/train/instance_segmentation.md)
- 4.[模型太大，想要更小的模型，试试模型裁剪吧!](tutorials/compress/classification.md)
