# 预测结果可视化

## 目录

* [paddlex.det.visualize](#1)
* [paddlex.seg.visualize](#2)
* [paddlex.visualize_det](#3)
* [paddlex.visualize_seg](#4)


## <h2 id="1">paddlex.det.visualize</h2>

```python
paddlex.det.visualize(image, result, threshold=0.5, save_dir='./', color=None)
```

> 将目标检测/实例分割模型预测得到的Box框和Mask在原图上进行可视化。

>
> **参数**
>
> > - **image** (str|np.ndarray): 原图文件路径或numpy数组(HWC排列，BGR格式)。
> > - **result** (str): 模型预测结果。
> > - **threshold** (float): score阈值，将Box置信度低于该阈值的框过滤不进行可视化。默认0.5
> > - **save_dir** (str): 可视化结果保存路径。若为None，则表示不保存，该函数将可视化的结果以np.ndarray的形式返回；若设为目录路径，则将可视化结果保存至该目录下。默认值为’./’。
> > - **color** (list|tuple|np.array): 各类别的BGR颜色值组成的数组，形状为Nx3（N为类别数量），数值范围为[0, 255]。例如针对2个类别的[[255, 0, 0], [0, 255, 0]]。若为None，则自动生成各类别的颜色。默认值为None。


使用示例：
```
import paddlex as pdx
model = pdx.load_model('xiaoduxiong_epoch_12')
result = model.predict('./xiaoduxiong_epoch_12/xiaoduxiong.jpeg')
pdx.det.visualize('./xiaoduxiong_epoch_12/xiaoduxiong.jpeg', result, save_dir='./')
# 预测结果保存在./visualize_xiaoduxiong.jpeg

```


## <h2 id="2">paddlex.seg.visualize</h2>

```python
paddlex.seg.visualize(image, result, weight=0.6, save_dir='./', color=None)
```

> 将语义分割模型预测得到的Mask在原图上进行可视化。

>
> **参数**
>
> > - **image** (str|np.ndarray): 原图文件路径或numpy数组(HWC排列，BGR格式)。
> > - **result** (str): 模型预测结果。
> > - **weight**(float): mask可视化结果与原图权重因子，weight表示原图的权重。默认0.6。
> > - **save_dir** (str): 可视化结果保存路径。若为None，则表示不保存，该函数将可视化的结果以np.ndarray的形式返回；若设为目录路径，则将可视化结果保存至该目录下。默认值为’./’。
> > - **color** (list): 各类别的BGR颜色值组成的列表。例如两类时可设置为[255, 255, 255, 0, 0, 0]。默认值为None，则使用默认生成的颜色列表。

使用示例：

```
import paddlex as pdx
model = pdx.load_model('cityscape_deeplab')
result = model.predict('city.png')
pdx.seg.visualize('city.png', result, save_dir='./')
# 预测结果保存在./visualize_city.png
```


## <h2 id="3">paddlex.visualize_det</h2>

> 是paddlex.det.visualize的别名，接口说明同 [paddlex.det.visualize](./visualize.md#paddlex.det.visualize)

## <h2 id="4">paddlex.visualize_seg</h2>

> 是paddlex.seg.visualize的别名，接口说明同 [paddlex.seg.visualize](./visualize.md#paddlex.seg.visualize)
