# 数据集工具

## 数据集分析

### paddlex.datasets.analysis.Seg
```python
paddlex.datasets.analysis.Seg(data_dir, file_list, label_list)
```

构建统计分析语义分类数据集的分析器。

> **参数**
> > * **data_dir** (str): 数据集所在的目录路径。  
> > * **file_list** (str): 描述数据集图片文件和类别id的文件路径（文本内每行路径为相对`data_dir`的相对路径）。  
> > * **label_list** (str): 描述数据集包含的类别信息文件路径。  

#### analysis
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

[代码示例](https://github.com/PaddlePaddle/PaddleX/blob/develop/examples/multi-channel_remote_sensing/tools/analysis.py)

[统计信息示例](../../examples/multi-channel_remote_sensing/analysis.html#id2)

#### cal_clipped_mean_std
```python
cal_clipped_mean_std(self, clip_min_value, clip_max_value, data_info_file)
```

Seg分析器用于计算图像截断后的均值和方差的接口。

> **参数**
> > * **clip_min_value** (list):  截断的下限，小于min_val的数值均设为min_val。
> > * **clip_max_value** (list): 截断的上限，大于max_val的数值均设为max_val。
> > * **data_info_file** (str): 在analysis()接口中保存的分析结果文件(名为`train_information.pkl`)的路径。

[代码示例](https://github.com/PaddlePaddle/PaddleX/blob/develop/examples/multi-channel_remote_sensing/tools/cal_clipped_mean_std.py)

[计算结果示例](../examples/multi-channel_remote_sensing/analysis.html#id4)

## 数据集生成

### paddlex.det.paste_objects
```python
paddlex.det.paste_objects(templates, background, save_dir='dataset_clone')
```

将目标物体粘贴在背景图片上生成新的图片和标注文件

> **参数**
> > * **templates** (list|tuple)：可以将多张图像上的目标物体同时粘贴在同一个背景图片上，因此templates是一个列表，其中每个元素是一个dict，表示一张图片的目标物体。一张图片的目标物体有`image`和`annos`两个关键字，`image`的键值是图像的路径，或者是解码后的排列格式为（H, W, C）且类型为uint8且为BGR格式的数组。图像上可以有多个目标物体，因此`annos`的键值是一个列表，列表中每个元素是一个dict，表示一个目标物体的信息。该dict包含`polygon`和`category`两个关键字，其中`polygon`表示目标物体的边缘坐标，例如[[0, 0], [0, 1], [1, 1], [1, 0]]，`category`表示目标物体的类别，例如'dog'。
> > * **background** (dict): 背景图片可以有真值，因此background是一个dict，包含`image`和`annos`两个关键字，`image`的键值是背景图像的路径，或者是解码后的排列格式为（H, W, C）且类型为uint8且为BGR格式的数组。若背景图片上没有真值，则`annos`的键值是空列表[]，若有，则`annos`的键值是由多个dict组成的列表，每个dict表示一个物体的信息，包含`bbox`和`category`两个关键字，`bbox`的键值是物体框左上角和右下角的坐标，即[x1, y1, x2, y2]，`category`表示目标物体的类别，例如'dog'。
> > * **save_dir** (str)：新图片及其标注文件的存储目录。默认值为`dataset_clone`。

> **代码示例**

```python
import paddlex as pdx
templates = [{'image': 'dataset/JPEGImages/budaodian-10.jpg',
              'annos': [{'polygon': [[146, 169], [909, 169], [909, 489], [146, 489]],
                        'category': 'lou_di'},
                        {'polygon': [[146, 169], [909, 169], [909, 489], [146, 489]],
                        'category': 'lou_di'}]}]
background = {'image': 'dataset/JPEGImages/budaodian-12.jpg', 'annos': []}
pdx.det.paste_objects(templates, background, save_dir='dataset_clone')
```
