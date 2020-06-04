# 通过负样本学习降低误检率

## 应用场景

在背景和目标相似的场景下，模型容易把背景误检成目标。为了降低误检率，可以通过负样本学习来降低误检率，即在训练过程中把无目标真值的图片加入训练。

## 效果对比

* 与基准模型相比，通过负样本学习后的模型**mmAP有3.6%的提升，mAP有0.1%的提升**。
* 与基准模型相比，通过负样本学习后的模型在背景图片上的图片级别**误检率降低了49.68%**。
* 与基准模型相比，通过负样本学习后的模型在目标图片上的图片级别**召回率仅降低了1.22%**。

表1 违禁品验证集上**框级别精度**对比

||mmAP（AP@IoU=0.5:0.95）| mAP (AP@IoU=0.5)|
|:---|:---|:---|
|基准模型 | 45.8% | 83% |
|通过负样本学习后的模型 | 49.4% | 83.1% |

表2 违禁品验证集上**图片级别的召回率**、无违禁品验证集上**图片级别的误检率**对比

||违禁品图片级别的召回率| 无违禁品图片级别的误检率|
|:---|:--------------------|:------------------------|
|基准模型 | 98.97% | 55.27% |
|通过负样本学习后的模型 | 97.75% | 5.59% |

【名词解释】

 * 图片级别的召回率：只要在有目标的图片上检测出目标（不论框的个数），该图片被认为召回。批量有目标图片中被召回图片所占的比例，即为图片级别的召回率。

 * 图片级别的误检率：只要在无目标的图片上检测出目标（不论框的个数），该图片被认为误检。批量无目标图片中被误检图片所占的比例，即为图片级别的误检率。


## 使用方法

在定义训练所用的数据集之后，使用数据集类的成员函数`add_negative_samples`将无目标真值的背景图片所在路径传入给训练集。代码示例如下：

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from paddlex.det import transforms
import paddlex as pdx

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=600, max_size=1000),
    transforms.Padding(coarsest_stride=32)
])
eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=600, max_size=1000),
    transforms.Padding(coarsest_stride=32),
])

# 定义训练所用的数据集
train_dataset = pdx.datasets.CocoDetection(
    data_dir='jinnan2_round1_train_20190305/restricted/',
    ann_file='jinnan2_round1_train_20190305/train.json',
    transforms=train_transforms,
    shuffle=True,
    num_workers=2)
# 训练集中加入无目标背景图片
train_dataset.add_negative_samples('jinnan2_round1_train_20190305/normal_train_back/')

# 定义验证所用的数据集
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='jinnan2_round1_train_20190305/restricted/',
    ann_file='jinnan2_round1_train_20190305/val.json',
    transforms=eval_transforms,
    num_workers=2)

# 初始化模型，并进行训练
model = pdx.det.FasterRCNN(num_classes=len(train_dataset.labels) + 1)
model.train(
    num_epochs=17,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    train_batch_size=8,
    learning_rate=0.01,
    lr_decay_epochs=[13, 16],
    save_dir='./output')
```

## 实验细则

(1) 数据集

我们使用X光违禁品数据集对通过负样本学习降低误检率的策略有效性进行了实验验证。该数据集中背景比较繁杂，很多背景物体与目标物体较为相似。

* 检测铁壳打火机、黑钉打火机 、刀具、电源和电池、剪刀5种违禁品。

* 训练集有883张违禁品图片，验证集有98张违禁品图片。

* 无违禁品的X光图片有2540张。

(2) 基准模型

使用FasterRCNN-ResNet50作为检测模型，除了水平翻转外没有使用其他的数据增强方式，只使用违禁品训练集进行训练。模型在违禁品验证集上的精度见表1，mmAP有45.8%，mAP达到83%。

(3) 通过负样本学习后的模型

把无违禁品的X光图片按1:1分成无违禁品训练集和无违禁品验证集。我们将基准模型在无违禁品验证集进行测试，发现图片级别的误检率高达55.27%。为了降低该误检率，将基准模型在无违禁品训练集进行测试，挑选出被误检图片共663张，将这663张图片加入训练，训练参数配置与基准模型训练时一致。

通过负样本学习后的模型在违禁品验证集上的精度见表1，mmAP有49.4%，mAP达到83.1%。与基准模型相比，**mmAP有3.6%的提升，mAP有0.1%的提升**。通过负样本学习后的模型在无违禁品验证集的误检率仅有5.58%，与基准模型相比，**误检率降低了49.68%**。

此外，还测试了两个模型在有违禁品验证集上图片级别的召回率，见表2，与基准模型相比，通过负样本学习后的模型仅漏检了1张图片，召回率几乎是无损的。
