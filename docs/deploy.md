# 模型部署导出

### 导出inference模型

在服务端部署的模型需要首先将模型导出为inference格式模型，导出的模型将包括`__model__`、`__params__`和`model.yml`三个文名，分别为模型的网络结构，模型权重和模型的配置文件（包括数据预处理参数等等）。在安装完PaddleX后，在命令行终端使用如下命令导出模型到当前目录`inferece_model`下。

> 可直接下载垃圾检测模型测试本文档的流程[garbage_epoch_12.tar.gz](https://bj.bcebos.com/paddlex/models/garbage_epoch_12.tar.gz)

```
paddlex --export_inference --model_dir=./garbage_epoch_12 --save_dir=./inference_model
```

## 模型C++和Python部署方案预计一周内推出...
