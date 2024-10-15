简体中文 | [English](config_parameters_common_en.md)

# PaddleX通用模型配置文件参数说明

# Global
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|model|str|指定模型名称|yaml文件中指定的模型名称|
|mode|str|指定模式（check_dataset/train/evaluate/export/predict）|check_dataset|
|dataset_dir|str|数据集路径|yaml文件中指定的数据集路径|
|device|str|指定使用的设备|yaml文件中指定的设备id|
|output|str|输出路径|"output"|
# CheckDataset
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|convert.enable|bool|是否进行数据集格式转换；图像分类、行人属性识别、车辆属性识别、文档方向分类、主体检测、行人检测、车辆检测、人脸检测、异常检测、文本检测、印章文本检测、文本识别、表格识别、图像矫正、版面区域检测暂不支持数据格式转换；图像多标签分类支持COCO格式的数据转换；图像特征、语义分割、实例分割支持LabelMe格式的数据转换；目标检测和小目标检测支持VOC、LabelMe格式的数据转换；公式识别支持PKL格式的数据转换；时序预测、时序异常检测、时序分类支持xlsx和xls格式的数据转换|False|
|convert.src_dataset_type|str|需要转换的源数据集格式|null|
|split.enable|bool|是否重新划分数据集|False|
|split.train_percent|int|设置训练集的百分比，类型为0-100之间的任意整数，需要保证和val_percent值加和为100；|null|
|split.val_percent|int|设置验证集的百分比，类型为0-100之间的任意整数，需要保证和train_percent值加和为100；|null|
|split.gallery_percent|int|设置验证集中被查询样本的百分比，类型为0-100之间的任意整数，需要保证和train_percent、query_percent，值加和为100；该参数只有图像特征模块才会使用|null|
|split.query_percent|int|设置验证集中查询样本的百分比，类型为0-100之间的任意整数，需要保证和train_percent、gallery_percent，值加和为100；该参数只有图像特征模块才会使用|null|

# Train
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|num_classes|int|数据集中的类别数；如果您需要在私有数据集进行训练，需要对该参数进行设置；图像矫正、文本检测、印章文本检测、文本识别、公式识别、表格识别、时序预测、时序异常检测、时序分类不支持该参数|yaml文件中指定类别数|
|epochs_iters|int|模型对训练数据的重复学习次数|yaml文件中指定的重复学习次数|
|batch_size|int|训练批大小|yaml文件中指定的训练批大小|
|learning_rate|float|初始学习率|yaml文件中指定的初始学习率|
|pretrain_weight_path|str|预训练权重路径|null|
|warmup_steps|int|预热步数|yaml文件中指定的预热步数|
|resume_path|str|模型中断后的恢复路径|null|
|log_interval|int|训练日志打印间隔|yaml文件中指定的训练日志打印间隔|
|eval_interval|int|模型评估间隔|yaml文件中指定的模型评估间隔|
|save_interval|int|模型保存间隔；异常检测、语义分割、图像矫正、时序预测、时序异常检测、时序分类暂不支持该参数|yaml文件中指定的模型保存间隔|

# Evaluate
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|weight_path|str|评估模型路径|默认训练产出的本地路径，当指定为None时，表示使用官方权重|
|log_interval|int|评估日志打印间隔|yaml文件中指定的评估日志打印间隔|
# Export
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|weight_path|str|导出模型的动态图权重路径|默认训练产出的本地路径，当指定为None时，表示使用官方权重|
# Predict
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|batch_size|int|预测批大小|yaml文件中指定的预测批大小|
|model_dir|str|预测模型路径|默认训练产出的本地推理模型路径，当指定为None时，表示使用官方权重|
|input|str|预测输入路径|yaml文件中指定的预测输入路径|
