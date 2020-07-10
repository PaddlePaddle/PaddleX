# 模型版本升级

由于PaddleX代码的持续更新，版本低于1.0.0的模型暂时无法直接用于预测部署，用户需要按照以下步骤对模型版本进行转换，转换后的模型可以在多端上完成部署。

## 检查模型版本

存放模型的文件夹存有一个`model.yml`文件，该文件的最后一行`version`值表示模型的版本号，若版本号小于1.0.0，则需要进行版本转换，若版本号大于及等于1.0.0，则不需要进行版本转换。

## 版本转换

```
paddlex --export_inference --model_dir=/path/to/low_version_model --save_dir=/path/to/high_version_model
```
`--model_dir`为版本号小于1.0.0的模型路径，可以是PaddleX训练过程保存的模型，也可以是导出为inference格式的模型。`--save_dir`为转换为高版本的模型，后续可用于多端部署。
