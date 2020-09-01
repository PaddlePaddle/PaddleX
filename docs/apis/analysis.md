# 数据集分析

## paddlex.datasets.analysis.Seg
```python
paddlex.datasets.analysis.Seg(data_dir, file_list, label_list)
```

构建统计分析语义分类数据集的分析器。

> **参数**
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和类别id的文件路径（文本内每行路径为相对`data_dir`的相对路径）。  
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  

### analysis
```python
analysis(self)
```

Seg分析器的分析接口，完成以下信息的分析统计：

> * 图像数量
> * 图像最大和最小的尺寸
> * 图像通道数量
> * 图像各通道的最小值和最大值
> * 图像各通道的像素值分布
> * 图像各通道归一化后的均值和方差
> * 标注图中各类别的数量及比重

[代码示例](https://github.com/PaddlePaddle/PaddleX/examples/multi-channel_remote_sensing/tools/analysis.py)

[统计信息示例](../../examples/multi-channel_remote_sensing/analysis.html#id2)

### cal_clipped_mean_std
```python
cal_clipped_mean_std(self, clip_min_value, clip_max_value, data_info_file)
```

Seg分析器用于计算图像截断后的均值和方差的接口。

> **参数**
> > * **clip_min_value** (list):  截断的下限，小于min_val的数值均设为min_val。
> > * **clip_max_value** (list): 截断的上限，大于max_val的数值均设为max_val。
> > * **data_info_file** (str): 在analysis()接口中保存的分析结果文件(名为`train_information.pkl`)的路径。

[代码示例](https://github.com/PaddlePaddle/PaddleX/examples/multi-channel_remote_sensing/tools/cal_clipped_mean_std.py)

[计算结果示例](../../examples/multi-channel_remote_sensing/analysis.html#id4)
