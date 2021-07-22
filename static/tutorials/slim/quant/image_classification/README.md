# 图像分类模型量化

在此目录下提供了MobileNetV2模型的量化示例，执行如下命令即可

## 第一步 量化模型
```
python mobilenetv2_quant.py
```
执行代码会自动下载模型和数据集

## 第二步 导出为PaddleLite模型

```
python paddlelite_export.py
```
执行此脚本前，需安装paddlelite，在python环境中`pip install paddlelite`即可
