# 可视化-visualize
PaddleX提供了一系列模型预测和结果分析的可视化函数。

## 目标检测/实例分割预测结果可视化
```
paddlex.det.visualize(image, result, threshold=0.5, save_dir='./')
```
将目标检测/实例分割模型预测得到的Box框和Mask在原图上进行可视化

### 参数
> * **image** (str): 原图文件路径。  
> * **result** (str): 模型预测结果。
> * **threshold**(float): score阈值，将Box置信度低于该阈值的框过滤不进行可视化。默认0.5
> * **save_dir**(str): 可视化结果保存路径。若为None，则表示不保存，该函数将可视化的结果以np.ndarray的形式返回；若设为目录路径，则将可视化结果保存至该目录下。默认值为'./'。

### 使用示例
> 点击下载如下示例中的[模型](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz)和[测试图片](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong.jpeg)
```
import paddlex as pdx
model = pdx.load_model('xiaoduxiong_epoch_12')
result = model.predict('xiaoduxiong.jpeg')
pdx.det.visualize('xiaoduxiong.jpeg', result, save_dir='./')
# 预测结果保存在./visualize_xiaoduxiong.jpeg
```

## 目标检测/实例分割准确率-召回率可视化
```
paddlex.det.draw_pr_curve(eval_details_file=None, gt=None, pred_bbox=None, pred_mask=None, iou_thresh=0.5, save_dir='./')
```
将目标检测/实例分割模型评估结果中各个类别的准确率和召回率的对应关系进行可视化，同时可视化召回率和置信度阈值的对应关系。

### 参数
> * **eval_details_file** (str): 模型评估结果的保存路径，包含真值信息和预测结果。默认值为None。
> * **gt** (list): 数据集的真值信息。默认值为None。
> * **pred_bbox** (list): 模型在数据集上的预测框。默认值为None。
> * **pred_mask** (list): 模型在数据集上的预测mask。默认值为None。
> * **iou_thresh** (float): 判断预测框或预测mask为真阳时的IoU阈值。默认值为0.5。
> * **save_dir** (str): 可视化结果保存路径。默认值为'./'。

**注意：**`eval_details_file`的优先级更高，只要`eval_details_file`不为None，就会从`eval_details_file`提取真值信息和预测结果做分析。当`eval_details_file`为None时，则用`gt`、`pred_mask`、`pred_mask`做分析。

### 使用示例
> 示例一：
点击下载如下示例中的[模型](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz)和[数据集](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_ins_det.tar.gz)
```
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32)
])

eval_dataset = pdx.datasets.CocoDetection(
    data_dir='xiaoduxiong_ins_det/JPEGImages',
    ann_file='xiaoduxiong_ins_det/val.json',
    transforms=eval_transforms)

model = pdx.load_model('xiaoduxiong_epoch_12')
metrics, evaluate_details = model.evaluate(eval_dataset, batch_size=1, return_details=True)
gt = evaluate_details['gt']
bbox = evaluate_details['bbox']
mask = evaluate_details['mask']

# 分别可视化bbox和mask的准召曲线
pdx.det.draw_pr_curve(gt=gt, pred_bbox=bbox, pred_mask=mask, save_dir='./xiaoduxiong')
```
预测框的各个类别的准确率和召回率的对应关系、召回率和置信度阈值的对应关系可视化如下：
![](./images/xiaoduxiong_bbox_pr_curve(iou-0.5).png)

预测mask的各个类别的准确率和召回率的对应关系、召回率和置信度阈值的对应关系可视化如下：
![](./images/xiaoduxiong_segm_pr_curve(iou-0.5).png)

> 示例二：
使用[yolov3_darknet53.py示例代码](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/detection/yolov3_darknet53.py)训练完成后，加载模型评估结果文件进行分析:

```
import paddlex as pdx
eval_details_file = 'output/yolov3_darknet53/best_model/eval_details.json'
pdx.det.draw_pr_curve(eval_details_file, save_dir='./insect')
```
预测框的各个类别的准确率和召回率的对应关系、召回率和置信度阈值的对应关系可视化如下：
![](./images/insect_bbox_pr_curve(iou-0.5).png)

## 语义分割预测结果可视化
```
paddlex.seg.visualize(image, result, weight=0.6, save_dir='./')
```
将语义分割模型预测得到的Mask在原图上进行可视化

### 参数
> * **image** (str): 原图文件路径。  
> * **result** (str): 模型预测结果。
> * **weight**(float): mask可视化结果与原图权重因子，weight表示原图的权重。默认0.6。
> * **save_dir**(str): 可视化结果保存路径。若为None，则表示不保存，该函数将可视化的结果以np.ndarray的形式返回；若设为目录路径，则将可视化结果保存至该目录下。默认值为'./'。

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
