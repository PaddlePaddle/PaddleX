# 部署模型导出

**注：所有涉及到模型部署，均需要参考本文档，进行部署模型导出**

## 目录

* [模型格式说明](#1)
  * [训练模型格式](#11)
  * [部署模型格式](#12)
* [部署模型导出](#2)


## <h2 id="1">模型格式说明</h2>

### <h3 id="11">训练模型格式</h3>

在使用**PaddleX 2.0训练保存的模型**文件夹中，主要包含四个文件：
- `model.pdopt`，训练模型参数的优化器
- `model.pdparams`，模型参数
- `model.yml`，模型的配置文件（包括预处理参数、模型定义等）
- `eval_details.json`，模型评估时的预测结果和真值

需要注意的是，训练保存的模型不能直接用于部署，需要导出成部署格式后才能用于部署。

### <h3 id="12">部署模型格式</h3>

在服务端部署模型时需要将训练过程中保存的模型导出为inference格式模型，使用**PaddleX 2.0导出的inference格式模型**包括五个文件：
- `model.pdmodel`，模型网络结构
- `model.pdiparams`，模型权重
- `model.pdiparams.info`，模型权重名称
- `model.yml`，模型的配置文件（包括预处理参数、模型定义等）
- `pipeline.yml`，可用于[PaddleX Manufacture SDK](https://github.com/PaddlePaddle/PaddleX/tree/develop/deploy/cpp/docs/manufacture_sdk)的流程配置文件


需要注意的是，**PaddleX 2.0版本导出的inference格式与1.x版本的有所不同，1.x版本的inference格式模型是**：
- `__model__`，模型网络结构
- `__params__`，模型权重
- `model.yml`，模型的配置文件（包括预处理参数、模型定义等）


## <h2 id="2">部署模型导出</h2>

> **检查你的模型文件夹**，如果里面是`model.pdparams`、`model.pdopt`和`model.yml`3个文件时，那么就需要按照下面流程进行模型导出:

在安装完PaddleX后，在命令行终端使用如下命令将训练好的模型导出为部署所需格式：

```commandline
paddlex --export_inference --model_dir=./output/deeplabv3p_r50vd/best_model/ --save_dir=./inference_model
```

在路径`./inference_model`下会生成一个名为`inference_model`的文件夹，包含`model.pdmodel`、`model.pdiparams`、`model.pdiparams.info`、`model.yml`和`pipeline.yml`五个文件。


| 参数 | 说明 |
| ---- | ---- |
| --export_inference | 是否将模型导出为用于部署的inference格式，指定即为True |
| --model_dir | 待导出的模型路径，例如是`output/deeplabv3p_r50vd/best_model/` |
| --save_dir | 导出的模型存储路径，例如是`./inference_model` |
| --fixed_input_shape | 固定导出模型的输入大小，默认值为None |

使用TensorRT预测时，需固定模型的输入大小，通过`--fixed_input_shape `来指定输入大小`[w,h]`或者是`[n,c,w,h]`。例如指定为`[224,224]`时，输入大小为`[-1,3,224,224]`；若想同时固定住输入的批量大小，可设置为`[1,3,224,224]`:

```commandline
paddlex --export_inference --model_dir=./output/deeplabv3p_r50vd/best_model/ --save_dir=./inference_model --fixed_input_shape=[224,224]
```

**注意**：
- 分类模型的固定输入大小请保持与训练时的输入大小一致。
- 检测模型中YOLO/PPYOLO系列请保存w与h一致，且为32的倍数大小；指定`--fixed_input_shape`时，RCNN类的w和h需为32的倍数大小。
- 指定[w,h]时，w和h中间逗号隔开，不允许存在空格等其他字符。
- 需要注意的是，w,h设得越大，模型在预测过程中所需要的耗时和内存/显存占用越高；设得太小，会影响模型精度。
