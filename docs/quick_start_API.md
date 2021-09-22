# PaddleX API开发模式快速上手
通过简洁易懂的Python API，在兼顾功能全面性、开发灵活性、集成方便性的基础上，给开发者最流畅的深度学习开发体验。

## 目录
- [快速安装](#快速安装)
    - [PaddleX 2.0.0安装](#PaddleX-200安装)
    - [PaddleX develop安装](#PaddleX-develop安装)
- [使用前置说明](#使用前置说明)
    - [PaddleX的模型训练](#PaddleX的模型训练)
    - [PaddleX的其他用法](#PaddleX的其他用法)
- [使用示例](#使用示例)
    - <a href=#安装PaddleX>安装PaddleX</a>
    - <a href=#准备蔬菜分类数据集>准备蔬菜分类数据集</a>
    - <a href=#定义训练验证图像处理流程transforms>定义训练/验证图像处理流程transforms</a>
    - <a href=#定义dataset加载图像分类数据集>定义dataset加载图像分类数据集</a>
    - <a href=#使用MoibleNetV3_small模型开始训练>使用MoibleNetV3_small模型开始训练</a>
    - <a href=#训练过程使用VisualDL查看训练指标变化>训练过程使用VisualDL查看训练指标变化</a>
    - <a href=#加载训练保存的模型预测>加载训练保存的模型预测</a>

## 快速安装
以下安装过程默认用户已安装好**paddlepaddle-gpu或paddlepaddle(版本大于或等于2.1.2)**，paddlepaddle安装方式参照[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/release/2.0.0/install/pip/windows-pip.html)

### PaddleX 2.0.0安装
**我们推荐大家先安装Anacaonda，而后在新建的conoda环境中使用上述pip安装方式**。Anaconda是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。使用Anaconda可以通过创建多个独立的Python环境，避免用户的Python环境安装太多不同版本依赖导致冲突。参考[Anaconda安装PaddleX文档](./appendix/anaconda_install.md)

- Linux / macOS 操作系统

使用pip安装方式安装2.0.0版本：

```commandline
pip install paddlex==2.0.0 -i https://mirror.baidu.com/pypi/simple
```

paddlepaddle已集成pycocotools包，但也有pycocotools无法随paddlepaddle成功安装的情况。因PaddleX依赖pycocotools包，如遇到pycocotools安装失败，可参照如下方式安装pycocotools：

```commandline
pip install cython  
pip install pycocotools
```

- Windows 操作系统
使用pip安装方式安装2.0.0版本：

```commandline
pip install paddlex==2.0.0 -i https://mirror.baidu.com/pypi/simple
```

因PaddleX依赖pycocotools包，Windows安装时可能会提示`Microsoft Visual C++ 14.0 is required`，从而导致安装出错，[点击下载VC build tools](https://go.microsoft.com/fwlink/?LinkId=691126)安装再执行如下pip命令
> 注意：安装完后，需要重新打开新的终端命令窗口

```commandline
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

### PaddleX develop安装

github代码会跟随开发进度不断更新，可以安装release/2.0.0分支的代码使用最新的功能，安装方式如下：

```commandline
git clone https://github.com/PaddlePaddle/PaddleX.git
git checkout develop
cd PaddleX
pip install -r requirements.txt
python setup.py install
```

如遇到pycocotools安装失败，参考[PaddleX 2.0.0安装](./install.md#paddlex-200安装)中介绍的解决方法。

## 使用前置说明

### PaddleX的模型训练

跟随以下3个步骤，即可快速完成训练代码开发:

| 步骤 |                  |说明             |
| :--- | :--------------- | :-------------- |
| 第1步| <a href="#定义训练验证图像处理流程transforms">定义transforms</a>  | 用于定义模型训练、验证、预测过程中，<br>输入图像的预处理和数据增强操作 |
| 第2步| <a href="#定义dataset加载图像分类数据集">定义datasets</a>  | 用于定义模型要加载的训练、验证数据集 |
| 第3步| <a href="#使用MoibleNetV3_small模型开始训练">定义模型开始训练</a> | 选择需要的模型，进行训练 |

> **注意**：不同模型的transforms、datasets和训练参数都有较大差异。可直接根据[模型训练教程](../tutorials/train)获取更多模型的训练代码。

### PaddleX的其它用法

- <a href="#训练过程使用VisualDL查看训练指标变化">使用VisualDL查看训练过程中的指标变化</a>
- <a href="#加载训练保存的模型预测">加载训练保存的模型进行预测</a>

## 使用示例

接下来展示如何通过PaddleX在一个小数据集上进行训练。示例代码源于Github [tutorials/train/image_classification/mobilenetv3_small.py](../tutorials/train/image_classification/mobilenetv3_small.py)，用户可自行下载至本地运行。用户也可前往[AIStudio在线项目示例](https://aistudio.baidu.com/aistudio/projectdetail/2159977)学习体验。

<a name="安装PaddleX"></a>
**1. 安装PaddleX**  

PaddleX的安装以及安装问题的解决可以参考PaddleX的[安装文档](./install.md)。

<a name="准备蔬菜分类数据集"></a>
**2. 准备蔬菜分类数据集**  

```commandline
wget https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz
tar xzvf vegetables_cls.tar.gz
```

<a name="定义训练验证图像处理流程transforms"></a>
**3. 定义训练/验证图像处理流程transforms**  

因为训练时加入了数据增强操作，因此在训练和验证过程中，模型的数据处理流程需要分别进行定义。如下所示，代码在`train_transforms`中加入了[RandomCrop](./apis/transforms/transforms.md#randomcrop)和[RandomHorizontalFlip](./apis/transforms/transforms.md#randomhorizontalflip)两种数据增强方式, 更多方法可以参考[数据增强文档](./apis/transforms/transforms.md)。

```python
from paddlex import transforms as T
train_transforms = T.Compose([
    T.RandomCrop(crop_size=224),
    T.RandomHorizontalFlip(),
    T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256),
    T.CenterCrop(crop_size=224),
    T.Normalize()
])
```

<a name="定义dataset加载图像分类数据集"></a>
**4. 定义`dataset`加载图像分类数据集**  

定义数据集，`pdx.datasets.ImageNet`表示读取ImageNet格式的分类数据集：

```python
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

<a name="使用MoibleNetV3_small模型开始训练"></a>
**5. 使用MobileNetV3_small模型开始训练**  

本文档中使用百度基于蒸馏方法得到的MobileNetV3预训练模型，模型结构与MobileNetV3一致，但精度更高。PaddleX内置了20多种分类模型，查阅[PaddleX 图像分类模型API](apis/models/classification.md#其它分类模型)了解更多分类模型。
```python
num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_small(num_classes=num_classes)

model.train(num_epochs=10,
            train_dataset=train_dataset,
            train_batch_size=32,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_dir='output/mobilenetv3_small',
            use_vdl=True)
```

<a name="训练过程使用VisualDL查看训练指标变化"></a>
**6. 训练过程使用VisualDL查看训练指标变化**  

训练过程中，模型在训练集和验证集上的指标均会以标准输出流形式输出到命令终端。当用户设定`use_vdl=True`时，也会使用VisualDL格式将指标打点到`save_dir`目录下的`vdl_log`文件夹，在终端运行如下命令启动visualdl并查看可视化的指标变化情况。
```commandline
visualdl --logdir output/mobilenetv3_small --port 8001
```
服务启动后，通过浏览器打开https://0.0.0.0:8001或https://localhost:8001 即可。

如果您使用的是AIStudio平台进行训练，不能通过此方式启动visualdl，请参考AIStudio VisualDL启动教程使用

<a name="加载训练保存的模型预测"></a>
**7. 加载训练保存的模型预测**  

模型在训练过程中，会每间隔一定轮数保存一次模型，在验证集上评估效果最好的一轮会保存在`save_dir`目录下的`best_model`文件夹。通过如下方式可加载模型，进行预测：

```python
import paddlex as pdx
model = pdx.load_model('output/mobilenetv3_small/best_model')
result = model.predict('vegetables_cls/bocai/100.jpg')
print("Predict Result: ", result)
```
预测结果输出如下,
```
Predict Result: Predict Result: [{'score': 0.9999393, 'category': 'bocai', 'category_id': 0}]
```
