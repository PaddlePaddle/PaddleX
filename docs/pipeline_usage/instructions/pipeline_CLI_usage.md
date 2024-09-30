简体中文 | [English](pipeline_CLI_usage_en.md)

# PaddleX模型产线CLI命令行使用说明

在使用CLI命令行进行模型产线快速推理前，请确保您已经按照[PaddleX本地安装教程](../../installation/installation.md)完成了PaddleX的安装。

## 一、使用示例

### 1. 快速体验

以图像分类产线为例，使用方式如下：

```bash
paddlex --pipeline image_classification \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --device gpu:0 \
        --save_path ./output/
```
只需一步就能完成推理预测并保存预测结果，相关参数说明如下：

* `pipeline`：模型产线名称或是模型产线配置文件的本地路径，如模型产线名“image_classification”，或模型产线配置文件路径“path/to/image_classification.yaml”；
* `input`：待预测数据文件路径，支持本地文件路径、包含待预测数据文件的本地目录、文件URL链接；
* `device`：用于设置模型推理设备，如为GPU设置则可以指定卡号，如“cpu”、“gpu:2”，当不传入时，如有GPU设置则使用GPU，否则使用CPU；
* `save_path`：预测结果的保存路径，当不传入时，则不保存预测结果；

### 2. 自定义产线配置

如需对产线配置进行修改，可获取配置文件后进行修改，仍以图像分类产线为例，获取配置文件方式如下：

```bash
paddlex --get_pipeline_config image_classification

# Please enter the path that you want to save the pipeline config file: (default `./`)
./configs/

# The pipeline config has been saved to: configs/image_classification.yaml
```

然后可修改产线配置文件`configs/image_classification.yaml`，如图像分类配置文件内容为：

```yaml
Global:
  pipeline_name: image_classification
  input: https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg

Pipeline:
  model: PP-LCNet_x0_5
  batch_size: 1
  device: "gpu:0"
```

在修改完成后，即可使用该配置文件进行模型产线推理预测，方式如下：

```bash
paddlex --pipeline configs/image_classification.yaml \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --save_path ./output/

# {'input_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': array([0.62817, 0.03729, 0.03262, 0.03247, 0.03196]), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
```
