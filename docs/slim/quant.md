# 模型量化

模型量化将模型的计算从浮点型转为整型，从而加速模型的预测计算速度，在移动端/边缘端设备上降低模型的体积。

> 注：量化后的模型，通过PaddleLite转换为PaddleLite部署的模型格式后，模型体积将会大幅压缩。如若量化后的模型仍是以服务端本地部署格式(文件包括__model__和__params__)，那么模型的文件大小是无法呈现参数变化情况的。

## 使用方法

PaddleX中已经将量化功能作为模型导出的一个API，代码使用方式如下，本示例代码和模型数据均可通过GitHub项目上代码[tutorials/slim/quant/image_classification](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/slim/quant/image_classification)获取得到
```
import paddlex as pdx
model = pdx.load_model('mobilenetv2_vegetables')
# 加载数据集用于量化
dataset = pdx.datasets.ImageNet(
                data_dir='vegetables_cls',
                file_list='vegetables_cls/train_list.txt',
                label_list='vegetables_cls/labels.txt',
                transforms=model.test_transforms)

# 开始量化
pdx.slim.export_quant_model(model, dataset, 
			  batch_size=4,
			  batch_num=5,
	                  save_dir='./quant_mobilenet', 
	                  cache_dir='./tmp')
```

在获取本示例代码后，执行如下命令即可完成量化和PaddleLite的模型导出
```
# 将mobilenetv2模型量化保存
python mobilenetv2_quant.py
# 将量化后的模型导出为PaddleLite部署格式
python paddlelite_export.py
```

## 量化效果

在本示例中，我们可以看到模型量化后的服务端部署模型格式`server_mobilenet`和`quant_mobilenet`两个目录中，模型参数大小并无变化。 但在使用PaddleLite导出后，`mobilenetv2.nb`和`mobilenetv2_quant.nb`大小分别为8.8M, 2.7M，压缩至原来的31%。
