# 移动端部署

PaddleX的移动端部署由PaddleLite实现，部署的流程如下，首先将训练好的模型导出为inference model，然后使用PaddleLite的python接口对模型进行优化，最后使用PaddleLite的预测库进行部署，
PaddleLite的详细介绍和使用可参考：[PaddleLite文档](https://paddle-lite.readthedocs.io/zh/latest/)

> PaddleX --> Inference Model --> PaddleLite Opt --> PaddleLite Inference

以下介绍如何将PaddleX导出为inference model，然后使用PaddleLite的OPT模块对模型进行优化：

step 1: 安装PaddleLite

```
pip install paddlelite
```

step 2: 将PaddleX模型导出为inference模型

参考[导出inference模型](deploy_server/deploy_python.html#inference)将模型导出为inference格式模型。
**注意：由于PaddleX代码的持续更新，版本低于1.0.0的模型暂时无法直接用于预测部署，参考[模型版本升级](./upgrade_version.md)对模型版本进行升级。**

step 3: 将inference模型转换成PaddleLite模型

```
python /path/to/PaddleX/deploy/lite/export_lite.py --model_dir /path/to/inference_model --save_file /path/to/lite_model --place place/to/run

```

|  参数   | 说明  |
|  ----  | ----  |
| model_dir  | 预测模型所在路径，包含"__model__", "__params__"文件 |
| save_file  | 模型输出的名称，默认为"paddlex.nb" |
| place  | 运行的平台，可选：arm|opencl|x86|npu|xpu|rknpu|apu |


step 4: 预测

Lite模型预测正在集成中，即将开源...
