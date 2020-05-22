# 移动端部署

step 1: 安装PaddleLite

```
pip install paddlelite
```

step 2: 将PaddleX模型导出为inference模型

参考[导出inference模型](deploy_server/deploy_python.html#inference)将模型导出为inference格式模型。
**注意：由于PaddleX代码的持续更新，版本低于1.0.0的模型暂时无法直接用于预测部署，参考[模型版本升级](../upgrade_version.md)对模型版本进行升级。**

step 3: 将inference模型转换成PaddleLite模型

```
python /path/to/PaddleX/deploy/lite/export_lite.py --model_path /path/to/inference_model --save_dir /path/to/onnx_model
```

`--model_path`用于指定inference模型的路径，`--save_dir`用于指定Lite模型的保存路径。

step 4: 预测

Lite模型预测正在集成中，即将开源...
