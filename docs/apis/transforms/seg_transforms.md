# paddlex.seg.transforms

This section describes the operation on data used for segmentation tasks. The [Compose](#compose) class can be used to combine image preprocessing/augmenter operations.


## Compose
```python
paddlex.seg.transforms. Compose(transforms)
```
The input data is operated on according to the data preprocessing/data augmenter list. [Usage Example](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/fast_scnn.py#L15)
### Parameters
* **transforms** (list): Data preprocessing/data augmenter list.


## RandomHorizontalFlip
```python
paddlex.seg.transforms. RandomHorizontalFlip(prob=0.5)
```
Flip the image horizontally with a certain probability. It is the data augmenter operation during model training.
### Parameters
* **prob** (float): The probability of a random level flip. It is 0.5 by default.


## RandomVerticalFlip
```python
paddlex.seg.transforms. RandomVerticalFlip(prob=0.1)
```
Flip the image vertically with a certain probability. It is the data augmenter operation during model training.
### Parameters
* **prob** (float): probability of a random vertical flip. The default value is 0.1.


## Resize
```python
paddlex.seg.transforms. Resize(target_size, interp='LINEAR')
```
Resizes the image (resize).

- When the target size (target_size) type is int, resize the image to [[target_size, target_size]] according to the interpolation method.
- When the target size (target_size) type is list or tuple, resize the image to target_size. The input for target_size should be [w, h] or (w, h) according to the interpolation method.
### Parameters
* **target_size** (int|list|tuple): target size
* **interp** (str): resize interpolation. It is corresponding to opencv interpolation. The available values are 'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4' and the default value is "LINEAR". []


## ResizeByLong
```python
paddlex.seg.transforms. ResizeByLong(long_size)
```
Resize the long side of the image to a fixed value and scale the short side proportionally.
### Parameters
* **long_size** (int): Size of the long side of the image after resize.


## ResizeRangeScaling
```python
paddlex.seg.transforms. ResizeRangeScaling(min_value=400, max_value=600)
```
Randomly resize the long side of the image to the specified range. Scale the short side proportionally. Perform the data augmenter operation during model training.
### Parameters
* **min_value** (int): The minimum value after the long side is resized. The default value is 400.
* **max_value** (int): The maximum value after resizing the long side of the image. Default value is 600.


## ResizeStepScaling
```python
paddlex.seg.transforms. ResizeStepScaling(min_scale_factor=0.75, max_scale_factor=1.25, scale_step_size=0.25)
```
Resize the image by a scale in scale_step_size. It varies randomly in the range [[min_scale_factor, max_scale_factor]]. It is the data augmenter operation during model training.
### Parameters
* **min_scale_factor**(float), resize the minimum scale. The default value is 0.75.
* **max_scale_factor** (float), maximal resize scale. The default value is 1.25.
* **scale_step_size** (float), resize scale range interval. The default value is 0.25.


## Normalize
```python
paddlex.seg.transforms. Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0])
```
Standardize the image.

1 Pixel value minus min_val 2 Pixel value divided by (max_val - min_val), normalized to interval [0]. 0, 1.0 .3The image is subtracted from the mean and divided by the standard deviation.

### Parameters
* **mean** (list): The mean value of the image data set. Default values are 0[.]5, 0.5, 0.5 .The length should be the same as the number of image channels.
* **std** (list): Standard deviation of the image dataset. Default values are 0[.]5, 0.5, 0.5 .The length should be the same as the number of image channels.
* **min_val** (list): Minimum value of the image dataset. The default value is 0, 0, 0[.]The length should be the same as the number of image channels.
* **max_val** (list): The maximum value of the image dataset. The default value is 255[.]0, 255.0, 255.0 .The length should be the same as the number of image channels.

## Padding
```python
paddlex.seg.transforms. Padding(target_size, im_padding_value=[127.5, 127.5, 127.5], label_padding_value=255)
```
Perform padding on an image or label image with padding direction right and down. Padding is applied to the image or label image according to the provided value.
### Parameters
* **target_size** (int|list|tuple): Size of the image after padding.
* **im_padding_value** (list): The value of the padding of the image. The default value is 127[.]5, 127.5, 127.5 .The length should be the same as the number of image channels.
* **label_padding_value** (int): The value of the label image padding. The default value is 255 (this parameter only needs to be set during training).


## RandomPaddingCrop
```python
paddlex.seg.transforms. RandomPaddingCrop(crop_size=512, im_padding_value=[127.5, 127.5, 127.5], label_padding_value=255)
```
Random cropping of images and labeled maps. Perform the padding operations when the desired crop size is larger than the original. It is the data augmenter operation during model training.
### Parameters
* **crop_size**(int|list|tuple): The size of the crop image. The default value is 512.
* **im_padding_value** (list): The value of the padding of the image. The default value is 127[.]5, 127.5, 127.5 .The length should be the same as the number of image channels.
* **label_padding_value** (int): The value of the label image padding. The default value is 255.


## RandomBlur
```python
paddlex.seg.transforms. RandomBlur(prob=0.1)
```
Gaussian blurring of the image with a certain probability. It is the data augmenter operation during model training.
### Parameters
* **prob** (float): Probability of image blurring. It is 0.1 by default.


## RandomRotate
```python
paddlex.seg.transforms. RandomRotate(rotate_range=15, im_padding_value=[127.5, 127.5, 127.5], label_padding_value=255)
```
Random rotation of images. It is the data augmenter operation during model training. At present, it supports multi-channel RGB images. For example, it supports image data of multiple RGB images after concatenate along the channel axis, but it does not support image data with the number of channels not multiples of 3.

Random rotation of images within the rotation range [[-rotate_range, rotate_range]]. It is synchronized when labeled images exist. You can apply the corresponding padding to the rotated and labelled images.
### Parameters
* **rotate_range** (float): The maximum rotation angle. The default value is 15 degrees.
* **im_padding_value** (list): The value of the padding of the image. The default value is 127[.]5, 127.5, 127.5 .The length should be the same as the number of image channels.
* **label_padding_value** (int): The value of the label image padding. The default value is 255.


## RandomScaleAspect
```python
paddlex.seg.transforms. RandomScaleAspect(min_scale=0.5, aspect_ratio=0.33)
```
Crop and resize back to the original size image and labeled image. It is the data augmenter operation during model training.

Cropping and resizing the image according to the area ratio and aspect ratio. When there is a labeled image, the operation is performed simultaneously.
### Parameters
* **min_scale** (float): the area ratio of the crop image to the original image, The value is 0 or 1. If the value is 0, return to the original image. The default value is 0.5.[]
* **aspect_ratio** (float): the aspect ratio range of the crop image. It is a non-negative value. When it is 0, return to the original image. The default value is 0.33.


## RandomDistort
```python
paddlex.seg.transforms. RandomDistort(brightness_range=0.5, brightness_prob=0.5, contrast_range=0.5, contrast_prob=0.5, saturation_range=0.5, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```
Random pixel content transformation of the image with a certain probability. It is the data augmenter operation in the model training. At present, it supports multi-channel RGB images. For example, it supports image data of multiple RGB images after concatenate along the channel axis, but it does not support image data with the number of channels not multiples of 3.

1 Randomize the operation order of the transformations. 2 Perform a random pixel content transformation in the range[-range, range] with a certain probability on the image in the order shown in Step 1.

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

## Clip
```python
paddlex.seg.transforms. Clip(min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0])
```
Clip data that is out of range on the image.

### Parameters
* **min_val** (list): the lower limit of the clip, any value smaller than min_val is set to min_val. The default value is 0.
* **max_val** (list): The upper limit of the crop, any value greater than max_val will be set to max_val. The default value is 255.0.

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

### Parameters
* **mode** (str): Transforms所处的阶段，包括`train', 'eval'或'test'
* **min_max_size**(list): 用于对图像进行resize，具体作用参见上述步骤。
* **train_crop_size** (list): 训练过程中随机裁剪原图用于训练，具体作用参见上述步骤。此参数仅在mode为`train`时生效。
* **mean** (list): 图像均值, 默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像方差，默认为[0.229, 0.224, 0.225]。
* **random_horizontal_flip**(bool): 数据增强，是否随机水平翻转图像，此参数仅在mode为`train`时生效。

### Add data enhancement methods
```python
ComposedSegTransforms.add_augmenters(augmenters)
```
> **参数**
>
> * **augmenters**(list): 数据增强方式列表

#### Example
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
