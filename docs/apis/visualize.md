# 可视化-visualize
PaddleX提供了一系列模型预测和结果分析的可视化函数。

## 目标检测/实例分割预测结果可视化
```
paddlex.det.visualize(image, result, threshold=0.5, save_dir='./')
```
将目标检测/实例分割模型预测得到的Box框和Mask在原图上进行可视化。

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
点击下载如下示例中的[模型](https://bj.bcebos.com/paddlex/models/insect_epoch_270.zip)和[数据集](https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz)

> 方式一：分析训练过程中保存的模型文件夹中的评估结果文件`eval_details.json`，例如[模型](https://bj.bcebos.com/paddlex/models/insect_epoch_270.zip)中的`eval_details.json`。
```
import paddlex as pdx
eval_details_file = 'insect_epoch_270/eval_details.json'
pdx.det.draw_pr_curve(eval_details_file, save_dir='./insect')
```
> 方式二：分析模型评估函数返回的评估结果。

```
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

model = pdx.load_model('insect_epoch_270')
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/val_list.txt',
    label_list='insect_det/labels.txt',
    transforms=model.eval_transforms)
metrics, evaluate_details = model.evaluate(eval_dataset, batch_size=8, return_details=True)
gt = evaluate_details['gt']
bbox = evaluate_details['bbox']
pdx.det.draw_pr_curve(gt=gt, pred_bbox=bbox, save_dir='./insect')
```

预测框的各个类别的准确率和召回率的对应关系、召回率和置信度阈值的对应关系可视化如下：
![](./images/insect_bbox_pr_curve(iou-0.5).png)

## 语义分割预测结果可视化
```
paddlex.seg.visualize(image, result, weight=0.6, save_dir='./')
```
将语义分割模型预测得到的Mask在原图上进行可视化。

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
利用此接口，可以分析在不同的`eval_metric_loss`参数下，模型被裁剪的比例情况。可视化结果纵轴为eval_metric_loss参数值，横轴为对应的模型被裁剪的比例。

### 参数
>* **model** (paddlex.cv.models): 使用PaddleX加载的模型。
>* **sensitivities_file** (str): 模型各参数在验证集上计算得到的参数敏感度信息文件。

### 使用示例
> 点击下载示例中的[模型](https://bj.bcebos.com/paddlex/models/vegetables_mobilenet.tar.gz)和[sensitivities_file](https://bj.bcebos.com/paddlex/slim_prune/mobilenetv2.sensitivities)
```
import paddlex as pdx
model = pdx.load_model('vegetables_mobilenet')
pdx.slim.visualize(model, 'mobilenetv2.sensitivities', save_dir='./')
# 可视化结果保存在./sensitivities.png
```

## 可解释性结果可视化
```
paddlex.interpret.visualize(img_file, 
                            model, 
                            dataset=None, 
                            algo='lime',
                            num_samples=3000, 
                            batch_size=50,
                            save_dir='./')
```
将模型预测结果的可解释性可视化，支持LIME和NormLIME两种可解释性算法。  
LIME表示与模型无关的局部可解释性，可以解释任何模型。LIME的思想是以输入样本为中心，在其附近的空间中进行随机采样，每个采样通过原模型得到新的输出，这样得到一系列的输入和对应的输出，LIME用一个简单的、可解释的模型（比如线性回归模型）来拟合这个映射关系，得到每个输入维度的权重，以此来解释模型。  
NormLIME则是利用一定数量的样本来出一个全局的解释。NormLIME会提前计算一定数量的测试样本的LIME结果，然后对相同的特征进行权重的归一化，这样来得到一个全局的输入和输出的关系。  

**注意：** 可解释性结果可视化目前只支持分类模型。

### 参数
>* **img_file** (str): 预测图像路径。
>* **model** (paddlex.cv.models): paddlex中的模型。
>* **dataset** (paddlex.datasets): 数据集读取器，默认为None。
>* **algo** (str): 可解释性方式，当前可选'lime'和'normlime'。
>* **num_samples** (int): LIME用于学习线性模型的采样数，默认为3000。
>* **batch_size** (int): 预测数据batch大小，默认为50。
>* **save_dir** (str): 可解释性可视化结果（保存为png格式文件）和中间文件存储路径。 

**注意：** `dataset`参数只有在`algo`为"normlime"的情况下才使用，`dataset`读取的是一个数据集，该数据集不宜过大，否则计算时间会较长，但应包含所有类别的数据。
### 使用示例
> 对预测可解释性结果可视化的过程可参见[代码](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/interpret/interpret.py)。
