# 部署模型导出

在服务端部署的模型需要首先将模型导出为inference格式模型，导出的模型将包括`__model__`、`__params__`和`model.yml`三个文名，分别为模型的网络结构，模型权重和模型的配置文件（包括数据预处理参数等等）。在安装完PaddleX后，在命令行终端使用如下命令导出模型到当前目录`inferece_model`下。
> 可直接下载小度熊分拣模型测试本文档的流程[xiaoduxiong_epoch_12.tar.gz](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz)

```
paddlex --export_inference --model_dir=./xiaoduxiong_epoch_12 --save_dir=./inference_model
```

使用TensorRT预测时，需指定模型的图像输入shape:[w,h]。
**注**：
- 分类模型请保持于训练时输入的shape一致。
- 指定[w,h]时，w和h中间逗号隔开，不允许存在空格等其他字符

```
paddlex --export_inference --model_dir=./xiaoduxiong_epoch_12 --save_dir=./inference_model --fixed_input_shape=[640,960]
```
