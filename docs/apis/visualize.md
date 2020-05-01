# 可视化-visualize
PaddleX提供了一系列模型预测和结果分析的可视化函数。

## 目标检测/实例分割预测结果可视化
```
paddlex.det.visualize(image, result, threshold=0.5, save_dir=None)
```
将目标检测/实例分割模型预测得到的Box框和Mask在原图上进行可视化

### 参数
> * **image** (str): 原图文件路径。  
> * **result** (str): 模型预测结果。
> * **threshold**(float): score阈值，将Box置信度低于该阈值的框过滤不进行可视化。默认0.5
> * **save_dir**(str): 可视化结果保存路径。若为None，则表示不保存，该函数将可视化的结果以np.ndarray的形式返回；若设为目录路径，则将可视化结果保存至该目录下

### 使用示例
> 点击下载如下示例中的[模型](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz)和[测试图片](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong.jpeg)
```
import paddlex as pdx
model = pdx.load_model('xiaoduxiong_epoch_12')
result = model.predict('xiaoduxiong.jpeg')
pdx.det.visualize('xiaoduxiong.jpeg', result, save_dir='./')
# 预测结果保存在./visualize_xiaoduxiong.jpeg
```

## 语义分割预测结果可视化
```
paddlex.seg.visualize(image, result, weight=0.6, save_dir=None)
```
将语义分割模型预测得到的Mask在原图上进行可视化

### 参数
> * **image** (str): 原图文件路径。  
> * **result** (str): 模型预测结果。
> * **weight**(float): mask可视化结果与原图权重因子，weight表示原图的权重。默认0.6
> * **save_dir**(str): 可视化结果保存路径。若为None，则表示不保存，该函数将可视化的结果以np.ndarray的形式返回；若设为目录路径，则将可视化结果保存至该目录下

### 使用示例
> 点击下载如下示例中的[模型](https://bj.bcebos.com/paddlex/models/cityscape_deeplab.tar.gz)和[测试图片](https://bj.bcebos.com/paddlex/datasets/city.png)
```
import paddlex as pdx
model = pdx.load_model('cityscape_deeplab')
result = model.predict('city.png')
pdx.det.visualize('city.png', result, save_dir='./')
# 预测结果保存在./visualize_city.png
```

## 模型裁剪比例可视化分析
```
paddlex.slim.visualize(model, sensitivities_file)
```
利用此接口，可以分析在不同的`eval_metric_loss`参数下，模型被裁剪的比例情况。可视化结果纵轴为eval_metric_loss参数值，横轴为对应的模型被裁剪的比例

### 参数
>* **model**: 使用PaddleX加载的模型
>* **sensitivities_file**: 模型各参数在验证集上计算得到的参数敏感度信息文件

### 使用示例
> 点击下载示例中的[模型](https://bj.bcebos.com/paddlex/models/vegetables_mobilenet.tar.gz)和[sensitivities_file](https://bj.bcebos.com/paddlex/slim_prune/mobilenetv2.sensitivities)
```
import paddlex as pdx
model = pdx.load_model('vegetables_mobilenet')
pdx.slim.visualize(model, 'mobilenetv2.sensitivities', save_dir='./')
# 可视化结果保存在./sensitivities.png
```
