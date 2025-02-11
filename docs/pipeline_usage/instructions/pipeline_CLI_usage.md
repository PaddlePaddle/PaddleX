---
comments: true
---

# PaddleX模型产线CLI（命令行）使用说明

在使用CLI（命令行）进行模型产线快速推理前，请确保您已经按照[PaddleX本地安装教程](../../installation/installation.md)完成了PaddleX的安装。

## 一、使用示例

### 1. 快速体验

以图像分类产线为例，使用方式如下：

```bash
paddlex --pipeline image_classification \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --device gpu:0 \
        --save_path ./output/ \
        --topk 5
```

只需一步就能完成推理预测并保存预测结果，相关参数说明如下：

* `pipeline`：模型产线名称或是模型产线配置文件的本地路径，如模型产线名 “image_classification”，或模型产线配置文件路径 “path/to/image_classification.yaml”；
* `input`：待预测数据文件路径，支持本地文件路径、包含待预测数据文件的本地目录、文件URL链接；
* `device`：用于设置模型推理设备，如为 GPU 则可以指定卡号，如 “cpu”、“gpu:2”，默认情况下，如有 GPU 设置则使用 0 号 GPU，否则使用 CPU；
* `save_path`：预测结果的保存路径，默认情况下，不保存预测结果；
* _`推理超参数`_：不同产线根据具体情况提供了不同的推理超参数设置，该参数优先级大于产线配置文件。对于图像分类产线，则支持通过 `topk` 参数设置输出的前 k 个预测结果。其他产线请参考对应的产线说明文档；

### 2. 自定义产线配置

如需对产线进行修改，可获取产线配置文件后进行修改，仍以图像分类产线为例，获取配置文件方式如下：

```bash
paddlex --get_pipeline_config image_classification

# Please enter the path that you want to save the pipeline config file: (default `./`)
./configs/

# The pipeline config has been saved to: configs/image_classification.yaml
```

然后可修改产线配置文件 `configs/image_classification.yaml`，如图像分类配置文件内容为：

```yaml
pipeline_name: image_classification

SubModules:
  ImageClassification:
    module_name: image_classification
    model_name: PP-LCNet_x0_5
    model_dir: null
    batch_size: 4
    device: "gpu:0"
    topk: 5
```

在修改完成后，即可使用该配置文件进行模型产线推理预测，方式如下：

```bash
paddlex --pipeline configs/image_classification.yaml \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --save_path ./output/

# {'input_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': array([0.62817, 0.03729, 0.03262, 0.03247, 0.03196]), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
```
