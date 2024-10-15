简体中文 | [English](config_parameters_time_series_en.md)

# PaddleX时序任务模型配置文件参数说明

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
|convert.enable|bool|是否进行数据集格式转换；时序预测、时序异常检测、时序分类支持xlsx和xls格式的数据转换|False|
|convert.src_dataset_type|str|需要转换的源数据集格式|null|
|split.enable|bool|是否重新划分数据集|False|
|split.train_percent|int|设置训练集的百分比，类型为0-100之间的任意整数，需要保证和val_percent值加和为100；|null|
|split.val_percent|int|设置验证集的百分比，类型为0-100之间的任意整数，需要保证和train_percent值加和为100；|null|

# Train
### 时序任务公共参数
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|epochs_iters|int|模型对训练数据的重复学习次数|yaml文件中指定的重复学习次数|
|batch_size|int|批大小|yaml文件中指定的批大小|
|learning_rate|float|初始学习率|yaml文件中指定的初始学习率|
|time_col|str|时间列，须结合自己的数据设置时间序列数据集的时间列的列名称。|yaml文件中指定的时间列|
|freq|str or int|频率，须结合自己的数据设置时间频率，如：1min、5min、1h。|yaml文件中指定的频率|
### 时序预测参数
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|target_cols|str|目标变量列，须结合自己的数据设置时间序列数据集的目标变量的列名称，可以为多个，多个之间用','分隔|OT|
|input_len|int|对于时序预测任务，该参数表示输入给模型的历史时间序列长度；输入长度建议结合实际场景及预测长度综合考虑，一般来说设置的越大，能够参考的历史信息越多，模型精度通常越高。|96|
|predict_len|int|希望模型预测未来序列的长度；预测长度建议结合实际场景综合考虑，一般来说设置的越大，希望预测的未来序列越长，模型精度通常越低。|96|
|patience|int|early stop机制参数，指在停止训练之前，容忍模型在验证集上的性能多少次连续没有改进；耐心值越大，一般训练时间越长。|10|
### 时序异常检测
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|input_len|int|对于时序异常检测任务，该参数表示输入给模型的时间序列长度，会按照该长度对时间序列切片，预测该长度下这一段时序序列是否有异常；输入长度建议结合实际场景考虑。如：输入长度为 96，则表示希望预测 96 个时间点是否有异常。|96|
|feature_cols|str|特征变量表示能够判断设备是否异常的相关变量，例如设备是否异常，可能与设备运转时的散热量有关。结合自己的数据，设置特征变量的列名称，可以为多个，多个之间用','分隔。|feature_0,feature_1|
|label_col|str|代表时序时间点是否异常的编号，异常点为 1，正常点为 0。|label|
### 时序分类
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|target_cols|str|用于判别类别的特征变量列，须结合自己的数据设置时间序列数据集的目标变量的列名称，可以为多个，多个之间用','分隔|dim_0,dim_1,dim_2|
|freq|str or int|频率，须结合自己的数据设置时间频率，如：1min、5min、1h。|1|
|group_id|str|一个群组编号表示的是一个时序样本，相同编号的时序序列组成一个样本。结合自己的数据设置指定群组编号的列名称, 如：group_id。| group_id|
|static_cov_cols|str|代表时序的类别编号列，同一个样本的标签相同。结合自己的数据设置类别的列名称，如：label。|label|
# Evaluate
|参数名|数据类型|描述|默认值|
|-|-|-|-|
|weight_path|str|评估模型路径|默认训练产出的本地路径，当指定为None时，表示使用官方权重|

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
