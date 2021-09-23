# PaddleClas部署模型导出

当前支持PaddleClas release/2.1分支导出的模型进行部署。本文档以ResNet50模型为例，讲述从release-2.1分支导出模型并用PaddleX 进行cpp部署整个流程。 PaddleClas相关详细文档可以查看[官网文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/README_cn.md)

### 1.获取PaddleClas源码

```sh
git clone https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
git checkout realease/2.1
```

### 2. 导出基于ImageNet数据的预训练模型

将预训练权重下载至`models目录`

```sh
mkdir models
cd models
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams
cd ..
```

使用`export_model.py`导出部署模型，注意在指定`pretrained_models`，**路径不用写到最终的后辍**

```sh
python tools/export_model.py --model ResNet50 \
                             --pretrained_model ./models/ResNet50_pretrained \
                             --output_path ./ResNet50_infer
```

导出的部署模型会保存在`./ResNet50_infer`目录，其结构如下

```
ResNet50
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```



需要注意的是，在导出的模型中，仅包含了模型的权重和模型结构，并没有像`PaddleDetection`或`PaddleSeg`在模型导出后，同时给出一个yaml配置文件来表明模型的预处理和类别信息等等，因此在本部署代码中为PaddleClas提供了一个在ImageNet数据上的模版yaml配置文件，用户可直接使用。

[点击获取模版yaml配置文件](../../../resources/resnet50_imagenet.yml)

如若你的分类模型在自定义数据集上训练得到，请注意相应修改这个模版中的相关配置信息

