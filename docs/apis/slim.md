# 模型压缩

## paddlex.slim.prune.analysis
> **计算参数敏感度**
```
paddlex.slim.prune.analysis(model, dataset, batch_size, save_file='model.sensi.data')
```
此函数接口与`paddlex.slim.cal_params_sensitivites`接口功能一致，仅修改了函数名，参数名，顺序和默认值，推荐使用此接口。

使用示例参考[教程-模型裁剪训练](https://github.com/PaddlePaddle/PaddleX/tree/release/1.3/tutorials/slim/prune)

## paddlex.slim.cal_params_sensitivities
> 此函数接口与`paddlex.slim.prune.analysis`功能一致，推荐使用`paddlex.slim.prune.analysis`接口  
> **计算参数敏感度**  
```
paddlex.slim.cal_params_sensitivities(model, save_file, eval_dataset, batch_size=8)
```
计算模型中可剪裁参数在验证集上的敏感度，并将敏感度信息保存至文件`save_file`
1. 获取模型中可剪裁卷积Kernel的名称。
2. 计算每个可剪裁卷积Kernel不同剪裁率下的敏感度。

【注意】卷积的敏感度是指按照剪裁率将模型剪裁后模型精度的损失。选择合适的敏感度，对应地也能确定最终模型需要剪裁的参数列表和各剪裁参数对应的剪裁率。  

[查看使用示例](https://github.com/PaddlePaddle/PaddleX/tree/release/1.3/tutorials/compress/classification/cal_sensitivities_file.py#L33)

**参数**

* **model** (paddlex.cls.models/paddlex.det.models/paddlex.seg.models): paddlex加载的模型。
* **save_file** (str): 计算的得到的sensetives文件存储路径。
* **eval_dataset** (paddlex.datasets): 评估数据集的读取器。
* **batch_size** (int): 评估时的batch_size大小。


## paddlex.slim.export_quant_model
> **导出量化模型**  
```
paddlex.slim.export_quant_model(model, test_dataset, batch_size=2, batch_num=10, save_dir='./quant_model', cache_dir='./temp')
```
导出量化模型，该接口实现了Post Quantization量化方式，需要传入测试数据集，并设定`batch_size`和`batch_num`。量化过程中会以数量为`batch_size` * `batch_num`的样本数据的计算结果为统计信息完成模型的量化。

**参数**

* **model**(paddlex.cls.models/paddlex.det.models/paddlex.seg.models): paddlex加载的模型。
* **test_dataset**(paddlex.dataset): 测试数据集。
* **batch_size**(int): 进行前向计算时的批数据大小。
* **batch_num**(int): 进行向前计算时批数据数量。
* **save_dir**(str): 量化后模型的保存目录。
* **cache_dir**(str): 量化过程中的统计数据临时存储目录。


**使用示例**

点击下载如下示例中的[模型](https://bj.bcebos.com/paddlex/models/vegetables_mobilenetv2.tar.gz)，[数据集](https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz)
```
import paddlex as pdx
model = pdx.load_model('vegetables_mobilenet')
test_dataset = pdx.datasets.ImageNet(
                    data_dir='vegetables_cls',
                    file_list='vegetables_cls/train_list.txt',
                    label_list='vegetables_cls/labels.txt',
                    transforms=model.eval_transforms)
pdx.slim.export_quant_model(model, test_dataset, save_dir='./quant_mobilenet')
```
