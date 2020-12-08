# paddlex.cls.transforms

This section describes the operations for image classification tasks. The [Compose](#compose) class can be used to combine image preprocessing/augmenter operations.

## Compose
```python
paddlex.cls.transforms. Compose(transforms)
```

The input data is operated by the data preprocessing/augmenter operator. [Usage Example](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py#L15)

> **Parameters**
>
> * **transforms** (list): Data preprocessing/data augmenter list.


## Normalize
```python
paddlex.cls.transforms. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Standardize the image.
1. Images are normalized to range [0].0, 1.0.
2. The image is subtracted from the mean and divided by the standard deviation.

### Parameters
* **mean** (list): The mean value of the image data set. Default values are 0 .[485, 0.456, and 0.406.]
* **std** (list): Standard deviation of the image dataset. Default values are 0[.]229, 0.224, 0.225.

## ResizeByShort
```python
paddlex.cls.transforms. ResizeByShort(short_size=256, max_size=-1)
```

Resizes the image according to the short edge of the image.
1. Get the length of the long and short edges of the image.
2. According to the ratio of short side and short_size, calculate the target length of the long side. At this time, the resize ratio of height and width is short_size/original short side length.
3. If max_size>0, adjust the resize ratio: if the target length of the long side is > max_size, the resize ratio of height and width is max_size/the length of the long edge of the original image.
4. Resize the image according to the resize ratio.

### Parameters
* **short_size** (int): The target length of the short side of the resized image. The default value is 256.
* **max_size** (int): Maximal limit of the length of the long side target. The default value is -1.

## CenterCrop
```python
paddlex.cls.transforms. CenterCrop(crop_size=224)
```

Diffusely prune a square with `crop_size` at the center of the image.
1. Calculates the start point of the pruning.
2. Prune the image.

### Parameters
* **crop_size** (int): The length of the target edge to be pruned. The default value is 224.

## RandomCrop
```python
paddlex.cls.transforms. RandomCrop(crop_size=224, lower_scale=0.08, lower_ratio=3. / 4, upper_ratio=4. / 3)
```

Random pruning of images, data augmenter operations for the model training.
1. Calculate the height and width of random pruning according to lower_scale, lower_ratio and upper_ratio.
2. Pick the starting point of the random pruning according to the height and width of the random pruning.
3. Prune the image.
4. Resize the pruning image to crop_size*crop_size.

### Parameters
* **crop_size** (int): The length of the target edge to be resized after random cropping. The default value is 224.
* **lower_scale** (float): Minimum limit of the ratio of the crop area to the original area. The default value is 0.08.
* **lower_ratio** (float):The minimum limit of the width change scale. The default value is 3. / 4.
* **upper_ratio** (float): Minimum limit for the width change ratio. The default value is 4. / 3.

## RandomHorizontalFlip
```python
paddlex.cls.transforms. RandomHorizontalFlip(prob=0.5)
```

Flip the image horizontally at random with a certain probability. It is the data augmenter operation during model training.

### Parameters
* **prob** (float): The probability of a random level flip. The default value is 0.5.

## RandomVerticalFlip
```python
paddlex.cls.transforms. RandomVerticalFlip(prob=0.5)
```

Vertically flip the image at random with a certain probability. It is the data augmenter operation in the model training.

### Parameters
* **prob** (float): probability of a random vertical flip. The default value is 0.5.

## RandomRotate
```python
paddlex.cls.transforms. RandomRotate(rotate_range=30, prob=0.5)
```

Rotate the image with probability in [-rotate_range, rotaterange] angle range. It is the data augmenter operation in the model training.

### Parameters
* **rotate_range** (int): The range of the rotation degree. The default value is 30.
* **prob** (float):The probability of random rotation. The default value is 0.5.

## RandomDistort
```python
paddlex.cls.transforms. RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5, saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```

Random pixel content transformation of the image with a certain probability. It is the data augmenter operation in the model training.
1. Randomize the operation order of the transformations.
2. Perform a random pixel content transformation in the range[-range, range] with a certain probability on the image in the order shown in Step 1.

[Note] This data augmenter must be used before the Normalize.

### Parameters
* **brightness_range** (float): the range of the brightness factor. The default value is 0.9.
* **brightness_prob** (float): The probability that the brightness is adjusted randomly. The default value is 0.5.
* **contrast_range** (float): The range of the contrast factor. The default value is 0.9.
* **contrast_prob** (float): The probability of randomly adjusting the contrast. The default value is 0.5.
* **saturation_range** (float): The range of the saturation factor. The default value is 0.9.
* **saturation_prob** (float): The probability of randomly adjusting the saturation. The default value is 0.5.
* **hue_range** (int): The range of the hue factor. The default value is 18.
* **hue_prob** (float): The probability of randomly adjusting the hue. The default value is 0.5.

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

### Parameters
* **mode** (str): Transforms所处的阶段，包括`train', 'eval'或'test'
* **crop_size** (int|list): 输入到模型里的图像大小，默认为[224, 224]（与原图大小无关，根据上述几个步骤，会将原图处理成该图大小输入给模型训练)
* **mean** (list): 图像均值, 默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像方差，默认为[0.229, 0.224, 0.225]。
* **random_horizontal_flip**(bool): 数据增强，是否以0，5的概率使用随机水平翻转增强，仅在model为'train'时生效，默认为True。底层实现采用[paddlex.cls.transforms.RandomHorizontalFlip](#randomhorizontalflip)

### Add data enhancement methods
```python
ComposedClsTransforms.add_augmenters(augmenters)
```
> **参数**
>
> * **augmenters**(list): 数据增强方式列表

#### Example
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
