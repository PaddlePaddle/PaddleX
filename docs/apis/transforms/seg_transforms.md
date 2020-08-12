# paddlex.seg.transforms

对用于分割任务的数据进行操作。可以利用[Compose](#compose)类将图像预处理/增强操作进行组合。


## Compose
```python
paddlex.seg.transforms.Compose(transforms)
```
根据数据预处理/数据增强列表对输入数据进行操作。[使用示例](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/fast_scnn.py#L15)
### 参数
* **transforms** (list): 数据预处理/数据增强列表。


## RandomHorizontalFlip
```python
paddlex.seg.transforms.RandomHorizontalFlip(prob=0.5)
```
以一定的概率对图像进行水平翻转，模型训练时的数据增强操作。
### 参数
* **prob** (float): 随机水平翻转的概率。默认值为0.5。


## RandomVerticalFlip
```python
paddlex.seg.transforms.RandomVerticalFlip(prob=0.1)
```
以一定的概率对图像进行垂直翻转，模型训练时的数据增强操作。
### 参数
* **prob**  (float): 随机垂直翻转的概率。默认值为0.1。


## Resize
```python
paddlex.seg.transforms.Resize(target_size, interp='LINEAR')
```
调整图像大小（resize）。

- 当目标大小（target_size）类型为int时，根据插值方式，
      将图像resize为[target_size, target_size]。
- 当目标大小（target_size）类型为list或tuple时，根据插值方式，
  将图像resize为target_size, target_size的输入应为[w, h]或（w, h）。
### 参数
* **target_size** (int|list|tuple): 目标大小
* **interp** (str): resize的插值方式，与opencv的插值方式对应，
可选的值为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']，默认为"LINEAR"。


## ResizeByLong
```python
paddlex.seg.transforms.ResizeByLong(long_size)
```
对图像长边resize到固定值，短边按比例进行缩放。
### 参数
* **long_size** (int): resize后图像的长边大小。


## ResizeRangeScaling
```python
paddlex.seg.transforms.ResizeRangeScaling(min_value=400, max_value=600)
```
对图像长边随机resize到指定范围内，短边按比例进行缩放，模型训练时的数据增强操作。
### 参数
* **min_value** (int): 图像长边resize后的最小值。默认值400。
* **max_value** (int): 图像长边resize后的最大值。默认值600。


## ResizeStepScaling
```python
paddlex.seg.transforms.ResizeStepScaling(min_scale_factor=0.75, max_scale_factor=1.25, scale_step_size=0.25)
```
对图像按照某一个比例resize，这个比例以scale_step_size为步长，在[min_scale_factor, max_scale_factor]随机变动,模型训练时的数据增强操作。
### 参数
* **min_scale_factor**（float), resize最小尺度。默认值0.75。
* **max_scale_factor** (float), resize最大尺度。默认值1.25。
* **scale_step_size** (float), resize尺度范围间隔。默认值0.25。


## Normalize
```python
paddlex.seg.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```
对图像进行标准化。

1.图像像素归一化到区间 [0.0, 1.0]。
2.对图像进行减均值除以标准差操作。
### 参数
* **mean** (list): 图像数据集的均值。默认值[0.5, 0.5, 0.5]。
* **std** (list): 图像数据集的标准差。默认值[0.5, 0.5, 0.5]。


## Padding
```python
paddlex.seg.transforms.Padding(target_size, im_padding_value=[127.5, 127.5, 127.5], label_padding_value=255)
```
对图像或标注图像进行padding，padding方向为右和下。根据提供的值对图像或标注图像进行padding操作。
### 参数
* **target_size** (int|list|tuple): padding后图像的大小。
* **im_padding_value** (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
* **label_padding_value** (int): 标注图像padding的值。默认值为255（仅在训练时需要设定该参数）。


## RandomPaddingCrop
```python
paddlex.seg.transforms.RandomPaddingCrop(crop_size=512, im_padding_value=[127.5, 127.5, 127.5], label_padding_value=255)
```
对图像和标注图进行随机裁剪，当所需要的裁剪尺寸大于原图时，则进行padding操作，模型训练时的数据增强操作。
### 参数
* **crop_size**（int|list|tuple): 裁剪图像大小。默认为512。
* **im_padding_value** (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
* **label_padding_value** (int): 标注图像padding的值。默认值为255。


## RandomBlur
```python
paddlex.seg.transforms.RandomBlur(prob=0.1)
```
以一定的概率对图像进行高斯模糊，模型训练时的数据增强操作。
### 参数
* **prob** (float): 图像模糊概率。默认为0.1。


## RandomRotate
```python
paddlex.seg.transforms.RandomRotate(rotate_range=15, im_padding_value=[127.5, 127.5, 127.5], label_padding_value=255)
```
对图像进行随机旋转，模型训练时的数据增强操作。

在旋转区间[-rotate_range, rotate_range]内，对图像进行随机旋转，当存在标注图像时，同步进行，
并对旋转后的图像和标注图像进行相应的padding。
### 参数
* **rotate_range** (float): 最大旋转角度。默认为15度。
* **im_padding_value** (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
* **label_padding_value** (int): 标注图像padding的值。默认为255。


## RandomScaleAspect
```python
paddlex.seg.transforms.RandomScaleAspect(min_scale=0.5, aspect_ratio=0.33)
```
裁剪并resize回原始尺寸的图像和标注图像，模型训练时的数据增强操作。

按照一定的面积比和宽高比对图像进行裁剪，并reszie回原始图像的图像，当存在标注图时，同步进行。
### 参数
* **min_scale**  (float)：裁取图像占原始图像的面积比，取值[0，1]，为0时则返回原图。默认为0.5。
* **aspect_ratio** (float): 裁取图像的宽高比范围，非负值，为0时返回原图。默认为0.33。


## RandomDistort
```python
paddlex.seg.transforms.RandomDistort(brightness_range=0.5, brightness_prob=0.5, contrast_range=0.5, contrast_prob=0.5, saturation_range=0.5, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```
以一定的概率对图像进行随机像素内容变换，模型训练时的数据增强操作。

1.对变换的操作顺序进行随机化操作。
2.按照1中的顺序以一定的概率对图像在范围[-range, range]内进行随机像素内容变换。  

【注意】该数据增强必须在数据增强Normalize之前使用。

### 参数
* **brightness_range** (float): 明亮度因子的范围。默认为0.5。
* **brightness_prob** (float): 随机调整明亮度的概率。默认为0.5。
* **contrast_range** (float): 对比度因子的范围。默认为0.5。
* **contrast_prob** (float): 随机调整对比度的概率。默认为0.5。
* **saturation_range** (float): 饱和度因子的范围。默认为0.5。
* **saturation_prob** (float): 随机调整饱和度的概率。默认为0.5。
* **hue_range** (int): 色调因子的范围。默认为18。
* **hue_prob** (float): 随机调整色调的概率。默认为0.5。

<!--
## ComposedSegTransforms
```python
paddlex.det.transforms.ComposedSegTransforms(mode, min_max_size=[400, 600], train_crop_shape=[769, 769], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], random_horizontal_flip=True)
```
语义分割DeepLab和UNet模型中已经组合好的数据处理流程，开发者可以直接使用ComposedSegTransforms，简化手动组合transforms的过程, 该类中已经包含了[RandomHorizontalFlip](#RandomHorizontalFlip)、[ResizeStepScaling](#ResizeStepScaling)、[RandomPaddingCrop](#RandomPaddingCrop)3种数据增强方式，你仍可以通过[add_augmenters函数接口](#add_augmenters)添加新的数据增强方式。  
ComposedSegTransforms共包括以下几个步骤：
 > 训练阶段：
> > 1. 随机对图像以0.5的概率水平翻转，若random_horizontal_flip为False，则跳过此步骤
> > 2. 按不同的比例随机Resize原图, 处理方式参考[paddlex.seg.transforms.ResizeRangeScaling](#resizerangescaling)。若min_max_size为None，则跳过此步骤
> > 3. 从原图中随机crop出大小为train_crop_size大小的子图，如若crop出来的图小于train_crop_size，则会将图padding到对应大小
> > 4. 图像归一化
 > 预测阶段：
> > 1. 将图像的最长边resize至(min_max_size[0] + min_max_size[1])//2, 短边按比例resize。若min_max_size为None，则跳过此步骤
> > 1. 图像归一化

### 参数
* **mode** (str): Transforms所处的阶段，包括`train', 'eval'或'test'
* **min_max_size**(list): 用于对图像进行resize，具体作用参见上述步骤。
* **train_crop_size** (list): 训练过程中随机裁剪原图用于训练，具体作用参见上述步骤。此参数仅在mode为`train`时生效。
* **mean** (list): 图像均值, 默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像方差，默认为[0.229, 0.224, 0.225]。
* **random_horizontal_flip**(bool): 数据增强，是否随机水平翻转图像，此参数仅在mode为`train`时生效。

### 添加数据增强方式
```python
ComposedSegTransforms.add_augmenters(augmenters)
```
> **参数**
> * **augmenters**(list): 数据增强方式列表

#### 使用示例
```
import paddlex as pdx
from paddlex.seg import transforms
train_transforms = transforms.ComposedSegTransforms(mode='train', train_crop_size=[512, 512])
eval_transforms = transforms.ComposedSegTransforms(mode='eval')

# 添加数据增强
import imgaug.augmenters as iaa
train_transforms.add_augmenters([
			transforms.RandomDistort(),
			iaa.blur.GaussianBlur(sigma=(0.0, 3.0))
])
```
上面代码等价于
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.Composed([
		transforms.RandomDistort(),
		iaa.blur.GaussianBlur(sigma=(0.0, 3.0)),
		# 上面2行为通过add_augmenters额外添加的数据增强方式
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.ResizeStepScaling(),
        transforms.PaddingCrop(crop_size=[512, 512]),
        transforms.Normalize()
])
eval_transforms = transforms.Composed([
        transforms.Normalize()
])
```
-->
