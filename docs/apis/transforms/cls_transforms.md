# paddlex.cls.transforms

对图像分类任务的数据进行操作。可以利用[Compose](#compose)类将图像预处理/增强操作进行组合。

## Compose
```python
paddlex.cls.transforms.Compose(transforms)
```

根据数据预处理/增强算子对输入数据进行操作。  [使用示例](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py#L15)

> **参数**
> * **transforms** (list): 数据预处理/数据增强列表。

## Normalize
```python
paddlex.cls.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

对图像进行标准化。  
1. 对图像进行归一化到区间[0.0, 1.0]。  
2. 对图像进行减均值除以标准差操作。

### 参数
* **mean** (list): 图像数据集的均值。默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像数据集的标准差。默认为[0.229, 0.224, 0.225]。

## ResizeByShort
```python
paddlex.cls.transforms.ResizeByShort(short_size=256, max_size=-1)
```

根据图像的短边调整图像大小（resize）。  
1. 获取图像的长边和短边长度。  
2. 根据短边与short_size的比例，计算长边的目标长度，此时高、宽的resize比例为short_size/原图短边长度。  
3. 如果max_size>0，调整resize比例：
   如果长边的目标长度>max_size，则高、宽的resize比例为max_size/原图长边长度。
4. 根据调整大小的比例对图像进行resize。

### 参数
* **short_size** (int): 调整大小后的图像目标短边长度。默认为256。
* **max_size** (int): 长边目标长度的最大限制。默认为-1。

## CenterCrop
```python
paddlex.cls.transforms.CenterCrop(crop_size=224)
```

以图像中心点扩散裁剪长宽为`crop_size`的正方形  
1. 计算剪裁的起始点。  
2. 剪裁图像。

### 参数
* **crop_size** (int): 裁剪的目标边长。默认为224。

## RandomCrop
```python
paddlex.cls.transforms.RandomCrop(crop_size=224, lower_scale=0.08, lower_ratio=3. / 4, upper_ratio=4. / 3)
```

对图像进行随机剪裁，模型训练时的数据增强操作。
1. 根据lower_scale、lower_ratio、upper_ratio计算随机剪裁的高、宽。
2. 根据随机剪裁的高、宽随机选取剪裁的起始点。
3. 剪裁图像。
4. 调整剪裁后的图像的大小到crop_size*crop_size。

### 参数
* **crop_size** (int): 随机裁剪后重新调整的目标边长。默认为224。
* **lower_scale** (float): 裁剪面积相对原面积比例的最小限制。默认为0.08。
* **lower_ratio** (float): 宽变换比例的最小限制。默认为3. / 4。
* **upper_ratio** (float): 宽变换比例的最大限制。默认为4. / 3。

## RandomHorizontalFlip
```python
paddlex.cls.transforms.RandomHorizontalFlip(prob=0.5)
```

以一定的概率对图像进行随机水平翻转，模型训练时的数据增强操作。

### 参数
* **prob** (float): 随机水平翻转的概率。默认为0.5。

## RandomVerticalFlip
```python
paddlex.cls.transforms.RandomVerticalFlip(prob=0.5)
```

以一定的概率对图像进行随机垂直翻转，模型训练时的数据增强操作。

### 参数
* **prob** (float): 随机垂直翻转的概率。默认为0.5。

## RandomRotate
```python
paddlex.cls.transforms.RandomRotate(rotate_range=30, prob=0.5)
```

以一定的概率对图像在[-rotate_range, rotaterange]角度范围内进行旋转，模型训练时的数据增强操作。

### 参数
* **rotate_range** (int): 旋转度数的范围。默认为30。
* **prob** (float): 随机旋转的概率。默认为0.5。

## RandomDistort
```python
paddlex.cls.transforms.RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5, saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```

以一定的概率对图像进行随机像素内容变换，模型训练时的数据增强操作。  
1. 对变换的操作顺序进行随机化操作。
2. 按照1中的顺序以一定的概率对图像在范围[-range, range]内进行随机像素内容变换。  

【注意】该数据增强必须在数据增强Normalize之前使用。

### 参数
* **brightness_range** (float): 明亮度因子的范围。默认为0.9。
* **brightness_prob** (float): 随机调整明亮度的概率。默认为0.5。
* **contrast_range** (float): 对比度因子的范围。默认为0.9。
* **contrast_prob** (float): 随机调整对比度的概率。默认为0.5。
* **saturation_range** (float): 饱和度因子的范围。默认为0.9。
* **saturation_prob** (float): 随机调整饱和度的概率。默认为0.5。
* **hue_range** (int): 色调因子的范围。默认为18。
* **hue_prob** (float): 随机调整色调的概率。默认为0.5。

<!--
## ComposedClsTransforms
```python
paddlex.cls.transforms.ComposedClsTransforms(mode, crop_size=[224, 224], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], random_horizontal_flip=True)
```
分类模型中已经组合好的数据处理流程，开发者可以直接使用ComposedClsTransforms，简化手动组合transforms的过程, 该类中已经包含了[RandomCrop](#RandomCrop)和[RandomHorizontalFlip](#RandomHorizontalFlip)两种数据增强方式，你仍可以通过[add_augmenters函数接口](#add_augmenters)添加新的数据增强方式。  
ComposedClsTransforms共包括以下几个步骤：
> 训练阶段：
> > 1. 随机从图像中crop一块子图，并resize成crop_size大小
> > 2. 将1的输出按0.5的概率随机进行水平翻转, 若random_horizontal_flip为False，则跳过此步骤
> > 3. 将图像进行归一化
> 验证/预测阶段：
> > 1. 将图像按比例Resize，使得最小边长度为crop_size[0] * 1.14
> > 2. 从图像中心crop出一个大小为crop_size的图像
> > 3. 将图像进行归一化

### 参数
* **mode** (str): Transforms所处的阶段，包括`train', 'eval'或'test'
* **crop_size** (int|list): 输入到模型里的图像大小，默认为[224, 224]（与原图大小无关，根据上述几个步骤，会将原图处理成该图大小输入给模型训练)
* **mean** (list): 图像均值, 默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像方差，默认为[0.229, 0.224, 0.225]。
* **random_horizontal_flip**(bool): 数据增强，是否以0，5的概率使用随机水平翻转增强，仅在model为'train'时生效，默认为True。底层实现采用[paddlex.cls.transforms.RandomHorizontalFlip](#randomhorizontalflip)

### 添加数据增强方式
```python
ComposedClsTransforms.add_augmenters(augmenters)
```
> **参数**
> * **augmenters**(list): 数据增强方式列表

#### 使用示例
```
import paddlex as pdx
from paddlex.cls import transforms
train_transforms = transforms.ComposedClsTransforms(mode='train', crop_size=[320, 320])
eval_transforms = transforms.ComposedClsTransforms(mode='eval', crop_size=[320, 320])

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
from paddlex.cls import transforms
train_transforms = transforms.Composed([
		transforms.RandomDistort(),
		iaa.blur.GaussianBlur(sigma=(0.0, 3.0)),
		# 上面两个为通过add_augmenters额外添加的数据增强方式
		transforms.RandomCrop(crop_size=320),
		transforms.RandomHorizontalFlip(prob=0.5),
		transforms.Normalize()
])
eval_transforms = transforms.Composed([
		transforms.ResizeByShort(short_size=int(320*1.14)),
		transforms.CenterCrop(crop_size=320),
		transforms.Normalize()
])
```
-->
