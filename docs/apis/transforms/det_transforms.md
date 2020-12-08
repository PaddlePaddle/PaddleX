# paddlex.det.transforms

This section describes the operation of data of the object detection/instance segmentation tasks. The [Compose](#compose) class can be used to combine image preprocessing/augmenter operations.

## Compose
```python
paddlex.det.transforms. Compose(transforms)
```

The input data is operated by the data preprocessing/augmenter operator. [Usage Example](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/yolov3_mobilenetv1.py#L15)

### Parameters
* **transforms** (list): Data preprocessing/data augmenter list.

## Normalize
```python
paddlex.det.transforms. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Standardize the image.
1. Normalizes the image to the interval [[0].0, 1.0].
2. The image is subtracted from the mean and divided by the standard deviation.

### Parameters
* **mean** (list): The mean value of the image data set. Default values are 0 .[485, 0.456, and 0.406.]
* **std** (list): Standard deviation of the image dataset. Default values are 0[.]229, 0.224, 0.225.

## ResizeByShort
```python
paddlex.det.transforms. ResizeByShort(short_size=800, max_size=1333)
```

Resizes the image according to the short edge of the image.
1. Get the length of the long and short edges of the image.
2. According to the ratio of short side and short_size, calculate the target length of the long side. At this time, the resize ratio of height and width is short_size/original short side length.
3. If max_size>0, adjust the resize ratio: if the target length of the long side is > max_size, the resize ratio of height and width is max_size/the length of the long edge of the original image.
4. Resize the image according to the resize ratio.

### Parameters
* **short_size** (int): The length of the short side object. The default value is 800.
* **max_size** (int): Maximal limit of the length of the long side target. The default value is 1333.

## Padding
```python
paddlex.det.transforms. Padding(coarsest_stride=1)
```

Multiples of padding the length and width of the image to the coarsest_stride. If the input image is [300], `640`, and the coarsest_stride is 32, the rightmost and the bottom of the image is padded with 0, and the final output image is [[320, 640], because 300 is not a multiple of 32.]
1. Returns directly if coarsest_stride is 1.
2. Calculate the difference between the width and the height and the nearest coarsest_stride multiple
3. Based on the calculated difference, padding is performed on the rightmost and lowest part of the image.

### Parameters
* **coarsest_stride** (int): the length and width of the filled image is a multiple of this parameter. The default value is 1.

## Resize
```python
paddlex.det.transforms. Resize(target_size=608, interp='LINEAR')
```

Resizes the image (resize).
* When the target size (target_size) type is int, resize the image to [[target_size, target_size]] according to the interpolation method.
* When the target size (target_size) type is list or tuple, resize the image to target_size according to the interpolation method. [Note] When the interpolation method is "RANDOM", one of the interpolation methods is randomly selected for resize. It is the data augmenter operation during model training.

### Parameters
* **target_size** (int/list/tuple): target length of short side. Default value is 608.
* **interp** (str): The interpolation mode of resize, corresponding to the interpolation of opencv, with the value range 'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'[.]The default value is "LINEAR".

## RandomHorizontalFlip
```python
paddlex.det.transforms. RandomHorizontalFlip(prob=0.5)
```

Flip the image horizontally at random with a certain probability. It is the data augmenter operation during model training.

### Parameters
* **prob** (float): The probability of a random level flip. The default value is 0.5.

## RandomDistort
```python
paddlex.det.transforms. RandomDistort(brightness_range=0.5, brightness_prob=0.5, contrast_range=0.5, contrast_prob=0.5, saturation_range=0.5, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```

Random pixel content transformation of the image with a certain probability. It is the data augmenter operation in the model training.
1. Randomize the operation order of the transformations.
2. Perform a random pixel content transformation in the range[-range, range] with a certain probability on the image in the order shown in Step 1.

[Note] This data augmenter must be used before the Normalize.

### Parameters
* **brightness_range** (float): the range of the brightness factor. The default value is 0.5.
* **brightness_prob** (float): The probability that the brightness is adjusted randomly. The default value is 0.5.
* **contrast_range** (float): The range of the contrast factor. The default value is 0.5.
* **contrast_prob** (float): The probability of randomly adjusting the contrast. The default value is 0.5.
* **saturation_range** (float): The range of the saturation factor. The default value is 0.5.
* **saturation_prob** (float): The probability of randomly adjusting the saturation. The default value is 0.5.
* **hue_range** (int): The range of the hue factor. The default value is 18.
* **hue_prob** (float): The probability of randomly adjusting the hue. The default value is 0.5.

## MixupImage
```python
paddlex.det.transforms. MixupImage(alpha=1.5, beta=1.5, mixup_epoch=-1)
```

Perform mixup operations on images. It is the data augmenter operation during model training. Currently, only the YOLOv3 model supports this transform. When the mixup field does not exist in label_info, return directly. Otherwise, perform the following operations:
1. The random factor is extracted from the random beta distribution.
2. The processing varies with different scenarios.
   * When factor >= 1.0, remove the mixup field in label_info and return it directly.
   * When the factor <= 0.0, return the mixup field in label_info directly. The field is removed from label_info.
   * For the rest, perform the following operations: (1) multiply the original image by the factor, multiply the mixup image by (1-factor), and superimpose the two results. (2) Splice the original image label box and the mixup image label box. (3) Splice original image label box category and mixup image label box category. (4) Multiply the original image label box mixing score by the factor, and multiply the mixup image label box mixing score by (1-factor), and superimpose the 2 results.
3. Update the augment_shape information in im_info.

### Parameters
* **alpha** (float): The lower limit of the random beta distribution. The default value is 1.5.
* **beta** (float): The upper limit of the random beta distribution. The default value is 1.5.
* **mixup_epoch** (int): Use mixup augmentation in the previous mixup_epoch round. This policy does not take effect when this parameter is -1. The default value is -1.

## RandomExpand Class
```python
paddlex.det.transforms. RandomExpand(ratio=4. , prob=0.5, fill_value=[123.675, 116.28, 103.53])
```

Randomly expand the image. It is the data augmenter operation during model training.
1. Randomly select the expansion ratio (expansion is performed only when the expansion ratio is greater than 1).
2. Calculate the size of the expanded image.
3. Initialize the image whose pixel value is the input fill-in value, and paste the original image randomly on this image.
4. Compute the position coordinates of the expanded real label box from the original image pasted position.
5. Compute the position of the real segmentation area after expansion based on the original image pasted position.

### Parameters
* **ratio** (float): The maximum ratio of image expansion. The default value is 4.0.
* **prob** (float): Probability of random expansion. The default value is 0.5.
* **fill_value** (list): The initial fill-in value of the expanded image (0-255). The default value is 123[.]675, 116.28, 103.53.

[Note] This data augmenter must be used before the data augmenter of Resize and ResizeByShort.

## RandomCrop
```python
paddlex.det.transforms. RandomCrop(aspect_ratio=[.5, 2. ], thresholds=[.0, .1, .3, .5, .7, .9], scaling=[.3, 1. ], num_attempts=50, allow_no_crop=True, cover_all_box=False)
```

Random crop image. It is the data augmenter operation during model training.
1. If allow_no_crop is True, add 'no_crop' to the thresholds.
2. Randomly disrupt the thresholds.
3. Traverse the elements in the thresholds: (1) If the current thresh is 'no_crop', return the original image and label information. (2) Randomly retrieve the values of aspect_ratio and scaling, and calculate the height, width and start point of the candidate cropping area. (3) Calculate the IoU of the real label box and the candidate cropping area. If all IoUs of the real label box are less than thresh, go to step 3. (4) If the cover_all_box is True and the IoU of the real label box is less than thresh, go to step 3. (5) Filter out the real label boxes located in the candidate cropping area. If the number of valid boxes is 0, go to step 3. Otherwise, go to step 4.
4. Convert the position coordinates of the valid true label box relative to the candidate cropping region.
5. Convert the position coordinates of the valid segmentation region relative to the candidate crop region.

[Note] This data augmenter must be used before the data augmenter of Resize and ResizeByShort.

### Parameters
* **aspect_ratio** (list): the range of the scaling of the cropping short edge, in the form of min and max. The default values are [.5, 2.]。 []
* **thresholds** (list): the list of IoU thresholds to determine whether the cropped candidate region is valid. The default values are .[0, .1, .3, .5, .7, .9 . ]
* **scaling** (list): the range of the cropping area relative to the original area, in the form of min and max. The default values are [.3, 1.]。 []
* **num_attempts** (int): The number of attempts before giving up on finding a valid crop area. The default value is 50.
* **allow_no_crop** (bool): Whether to allow no cropping. It is true by default.
* **cover_all_box** (bool): whether or not require all real label boxes to be in the crop area. It is false by default.

<!--
## ComposedRCNNTransforms
```python
paddlex.det.transforms.ComposedRCNNTransforms(mode, min_max_size=[224, 224], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], random_horizontal_flip=True)
```
目标检测FasterRCNN和实例分割MaskRCNN模型中已经组合好的数据处理流程，开发者可以直接使用ComposedRCNNTransforms，简化手动组合transforms的过程, 该类中已经包含了[RandomHorizontalFlip](#RandomHorizontalFlip)数据增强方式，你仍可以通过[add_augmenters函数接口](#add_augmenters)添加新的数据增强方式。  
ComposedRCNNTransforms共包括以下几个步骤：
> 训练阶段：
> > 1. 随机以0.5的概率将图像水平翻转, 若random_horizontal_flip为False，则跳过此步骤
> > 2. 将图像进行归一化
> > 3. 图像采用[ResizeByShort](#ResizeByShort)方式，根据min_max_size参数，进行缩入
> > 4. 使用[Padding](#Padding)将图像的长和宽分别Padding成32的倍数
> 验证/预测阶段：
> > 1. 将图像进行归一化
> > 2. 图像采用[ResizeByShort](#ResizeByShort)方式，根据min_max_size参数，进行缩入
> > 3. 使用[Padding](#Padding)将图像的长和宽分别Padding成32的倍数

### 参数
* **mode** (str): Transforms所处的阶段，包括`train', 'eval'或'test'
* **min_max_size** (list): 输入模型中图像的最短边长度和最长边长度，参考[ResizeByShort](#ResizeByShort)（与原图大小无关，根据上述几个步骤，会将原图处理成相应大小输入给模型训练)，默认[800, 1333]
* **mean** (list): 图像均值, 默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像方差，默认为[0.229, 0.224, 0.225]。
* **random_horizontal_flip**(bool): 数据增强，是否以0.5的概率使用随机水平翻转增强，仅在mode为'train'时生效，默认为True。底层实现采用[paddlex.det.transforms.RandomHorizontalFlip](#randomhorizontalflip)

### 添加数据增强方式
```python
ComposedRCNNTransforms.add_augmenters(augmenters)
```
> **参数**
> * **augmenters**(list): 数据增强方式列表

#### 使用示例
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.ComposedRCNNTransforms(mode='train', min_max_size=[800, 1333])
eval_transforms = transforms.ComposedRCNNTransforms(mode='eval', min_max_size=[800, 1333])

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
		# 上面两个为通过add_augmenters额外添加的数据增强方式
		transforms.RandomHorizontalFlip(prob=0.5),
		transforms.Normalize(),
        transforms.ResizeByShort(short_size=800, max_size=1333),
        transforms.Padding(coarsest_stride=32)
])
eval_transforms = transforms.Composed([
		transforms.Normalize(),
        transforms.ResizeByShort(short_size=800, max_size=1333),
        transforms.Padding(coarsest_stride=32)
])
```


## ComposedYOLOv3Transforms
```python
paddlex.det.transforms.ComposedYOLOv3Transforms(mode, shape=[608, 608], mixup_epoch=250, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], random_distort=True, random_expand=True, random_crop=True, random_horizontal_flip=True)
```
目标检测YOLOv3模型中已经组合好的数据处理流程，开发者可以直接使用ComposedYOLOv3Transforms，简化手动组合transforms的过程, 该类中已经包含了[MixupImage](#MixupImage)、[RandomDistort](#RandomDistort)、[RandomExpand](#RandomExpand)、[RandomCrop](#RandomCrop)、[RandomHorizontalFlip](#RandomHorizontalFlip)5种数据增强方式，你仍可以通过[add_augmenters函数接口](#add_augmenters)添加新的数据增强方式。  
ComposedYOLOv3Transforms共包括以下几个步骤：
> 训练阶段：
> > 1. 在前mixup_epoch轮迭代中，使用MixupImage策略，若mixup_epoch为-1，则跳过此步骤
> > 2. 对图像进行随机扰动，包括亮度，对比度，饱和度和色调，若random_distort为False，则跳过此步骤
> > 3. 随机扩充图像，若random_expand为False， 则跳过此步骤
> > 4. 随机裁剪图像，若random_crop为False， 则跳过此步骤
> > 5. 将4步骤的输出图像Resize成shape参数的大小
> > 6. 随机0.5的概率水平翻转图像，若random_horizontal_flip为False，则跳过此步骤
> > 7. 图像归一化
> 验证/预测阶段：
> > 1. 将图像Resize成shape参数大小
> > 2. 图像归一化

### 参数
* **mode** (str): Transforms所处的阶段，包括`train', 'eval'或'test'
* **shape** (list): 输入模型中图像的大小（与原图大小无关，根据上述几个步骤，会将原图处理成相应大小输入给模型训练)， 默认[608, 608]
* **mixup_epoch**(int): 模型训练过程中，在前mixup_epoch轮迭代中，使用mixup策略，如果为-1，则不使用mixup策略， 默认250。底层实现采用[paddlex.det.transforms.MixupImage](#mixupimage)
* **mean** (list): 图像均值, 默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像方差，默认为[0.229, 0.224, 0.225]。
* **random_distort**(bool): 数据增强，是否在训练过程中随机扰动图像，仅在mode为'train'时生效，默认为True。底层实现采用[paddlex.det.transforms.RandomDistort](#randomdistort)
* **random_expand**(bool): 数据增强，是否在训练过程随机扩张图像，仅在mode为'train'时生效，默认为True。底层实现采用[paddlex.det.transforms.RandomExpand](#randomexpand)
* **random_crop**(bool): 数据增强，是否在训练过程中随机裁剪图像，仅在mode为'train'时生效，默认为True。底层实现采用[paddlex.det.transforms.RandomCrop](#randomcrop)
* **random_horizontal_flip**(bool): 数据增强，是否在训练过程中随机水平翻转图像，仅在mode为'train'时生效，默认为True。底层实现采用[paddlex.det.transforms.RandomHorizontalFlip](#randomhorizontalflip)

### 添加数据增强方式
```python
ComposedYOLOv3Transforms.add_augmenters(augmenters)
```
> **参数**
> * **augmenters**(list): 数据增强方式列表

#### 使用示例
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.ComposedYOLOv3Transforms(mode='train', shape=[480, 480])
eval_transforms = transforms.ComposedYOLOv3Transforms(mode='eval', shape=[480, 480])

# 添加数据增强
import imgaug.augmenters as iaa
train_transforms.add_augmenters([
			iaa.blur.GaussianBlur(sigma=(0.0, 3.0))
])
```
上面代码等价于
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.Composed([
		iaa.blur.GaussianBlur(sigma=(0.0, 3.0)),
		# 上面为通过add_augmenters额外添加的数据增强方式
        transforms.MixupImage(mixup_epoch=250),
        transforms.RandomDistort(),
        transforms.RandomExpand(),
        transforms.RandomCrop(),
        transforms.Resize(target_size=480, interp='RANDOM'),
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.Normalize()
])
eval_transforms = transforms.Composed([
        transforms.Resize(target_size=480, interp='CUBIC'),
		transforms.Normalize()
])
```
-->
