# Model Saving

## Training process preservation

PaddleX在模型训练过程中，根据`train`函数接口中的`save_interval_epoch`参数设置，每间隔相应轮数保存一次模型，模型目录中包含了`model.pdparams`, `model.yml`等文件。

在训练过程中保存的模型，可用于作为pretrain_weights继续训练模型，也可使用`paddlex.load_model`接口加载测试模型的预测和评估等。

## Deployment model export

在前面提到的训练中保存的模型，如若要用于部署（部署可参阅PaddleX文档中的模型多端部署章节），需导出为部署的模型格式，部署的模型目录中包含`__model__`，`__params__`和`model.yml`三个文件。

模型部署在Python层面，可以使用基于高性能预测库的python接口`paddlex.deploy.Predictor`，也可使用`paddlex.load_model`接口。

模型部署可参考文档[部署模型导出](../deploy/export_model.md)

> 【总结】如若模型目录中包含`model.pdparams`，那说明模型是训练过程中保存的，部署时需要进行导出；部署的模型目录中需包含`__model__`，`__params__`和`model.yml`三个文件。

## Model deployment file description

- `__model__`：保存了模型的网络结构信息
- `__params__`： 保存了模型网络中的参数权重
- `model.yml`：在PaddleX中，将模型的预处理，后处理，以及类别相关信息均存储在此文件中

## The model is exported to ONNX Model

PaddleX作为开放开源的套件，其中的大部分模型均支持导出为ONNX协议，满足开发者多样性的需求。
> 需要注意的是ONNX存在多个OpSet版本，下表为PaddleX各模型支持导出的ONNX协议版本。

| 模型 | ONNX OpSet 9 | ONNX OpSet 10 | ONNX OpSet 11 |
| :--- | :----------- | :-----------  | :------------ |
| 图像分类 | 支持 |  支持 | 支持 |
| 目标检测（仅YOLOv3系列) | - | 支持 | 支持 |
| 语义分割（FastSCNN不支持) | - | - | 支持 |

### How to export

- 1. 参考文档[部署模型导出](../deploy/eport_model.md)，将训练保存的模型导出为部署模型  
- 2. 安装paddle2onnx `pip install paddle2onnx`，转换命令如下，通过`--opset_version`指定版本(9/10/11)，转换使用方法参考[Paddle2ONNX说明](https://github.com/PaddlePaddle/paddle2onnx)

- 附: Paddle2ONNX参阅 [https://github.com/PaddlePaddle/paddle2onnx](https://github.com/PaddlePaddle/paddle2onnx)
