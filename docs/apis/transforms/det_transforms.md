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
For the combined data processing process in the target detection FasterRCNN and instance segmentation MaskRCNN models, developers can directly use ComposedRCNNTransforms to simplify the process of manually combining transforms. This class already contains the [RandomHorizontalFlip](#RandomHorizontalFlip) data enhancement method, and you can still add new data enhancement methods through the [add_augmenters function interface](#add_augmenters).  
ComposedRCNNTransforms includes the following steps:

> Training stage:
> > 1. Randomly flip the image horizontally with a probability of 0.5. If random_horizon_flip is False, skip this step
> > 2. The image was normalized
> > 3. According to the min_max_size parameter, the image is indented in [ResizeByShort](#ResizeByShort) mode
> > 4. Use [Padding](#Padding) to pad the length and width of the image to a multiple of 32
> Validation / Prediction stage:
> > 1. The image was normalized
> > 2. According to the min_max_size parameter, the image is indented in [ResizeByShort](#ResizeByShort) mode
> > 3. Use [Padding](#Padding) to pad the length and width of the image to a multiple of 32

### Parameters
* **mode** (str): The stage of Transforms, including 'train', 'eval' or 'test'
* **min_max_size** (list): Input the shortest edge length and longest edge length of the image in the model, refer to [ResizeByShort](#ResizeByShort)（It has nothing to do with the size of the original image. According to the above steps, the original image will be processed into the corresponding size and input to the model training). The default is [800, 1333]
* **mean** (list): Mean value of the image, the default is [0.485, 0.456, 0.406]。
* **std** (list): Variance of the image, the default is [0.229, 0.224, 0.225]。
* **random_horizontal_flip**(bool): Whether to use random horizontal flip enhancement with a probability of 0,5 is only effective when the model is' train '. The default value is true.The underlying implementation adopts[paddlex.det.transforms.RandomHorizontalFlip](#randomhorizontalflip)

### Add data enhancement methods
```python
ComposedRCNNTransforms.add_augmenters(augmenters)
```
> **参数**
>
> * **augmenters**(list): List of data enhancement methods

#### Example
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.ComposedRCNNTransforms(mode='train', min_max_size=[800, 1333])
eval_transforms = transforms.ComposedRCNNTransforms(mode='eval', min_max_size=[800, 1333])

# Add data enhancement
import imgaug.augmenters as iaa
train_transforms.add_augmenters([
			transforms.RandomDistort(),
			iaa.blur.GaussianBlur(sigma=(0.0, 3.0))
])
```
The code above is equivalent to
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.Composed([
		transforms.RandomDistort(),
		iaa.blur.GaussianBlur(sigma=(0.0, 3.0)),
		# The above two are additional data enhancement methods that are added through add_aummenters
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
For the data processing process that has been combined in the target detection YOLOv3 model, developers can directly use ComposedYOLOv3Transforms to simplify the process of manually combining transforms. This class already contains five data enhancement methods [MixupImage](#MixupImage)、[RandomDistort](#RandomDistort)、[RandomExpand](#RandomExpand)、[RandomCrop](#RandomCrop)、[RandomHorizontalFlip](#RandomHorizontalFlip). You can still add new data enhancement methods through the [add_augmenters function interface](#add_augmenters).  
ComposedYOLOv3Transforms includes the following steps:

> Training stage:
> > 1. In the previous mixup_epoch round iterations, use the MixupImage strategy. If mixup_epoch is - 1, skip this step
> > 2. Randomly disturb the image, including brightness, contrast, saturation and hue. If random_distort is False, skip this step
> > 3. Expand the image randomly. If random_expand is False, skip this step
> > 4. Crop the image randomly. If random_crop is false, skip this step
> > 5. Resize the output image of step 4 to the size of the shape parameter
> > 6. Randomly flip the image horizontally with a probability of 0.5. If random_horizon_flip is False, skip this step
> > 7. Image normalization
> Validation / Prediction stage:
> > 1. Resize the image to the shape parameter size
> > 2. Image normalization

### Parameters
* **mode** (str): The stage of Transforms, including 'train', 'eval' or 'test'
* **shape** (list):  The image size input into the model, the default is [608, 608]（It has nothing to do with the size of the original image. According to the above steps, the original image will be processed into the size of the graph and input to the model training)
* **mixup_epoch**(int): In the process of model training, the mixup strategy is used in the previous mixup_epoch iterations. If it is - 1, the mixup strategy is not used, and the default value is 250.The underlying implementation adopts [paddlex.det.transforms.MixupImage](#mixupimage)
* **mean** (list): Mean value of the image, the default is [0.485, 0.456, 0.406]。
* **std** (list):  Variance of the image, the default is [0.229, 0.224, 0.225]。
* **random_distort**(bool): Data enhancement, whether the image is randomly disturbed during the training process, only takes effect when the mode is 'train', and the default is true. The underlying implementation adopts [paddlex.det.transforms.RandomDistort](#randomdistort)
* **random_expand**(bool): Data enhancement, whether to expand images randomly during training, only takes effect when mode is' train '. The default value is true. The underlying implementation adopts [paddlex.det.transforms.RandomExpand](#randomexpand)
* **random_crop**(bool):Data enhancement, whether to crop images randomly during training, only takes effect when mode is' train '. The default is true. The underlying implementation adopts [paddlex.det.transforms.RandomCrop](#randomcrop)
* **random_horizontal_flip**(bool): Data enhancement, whether to flip the image horizontally randomly during the training process, only takes effect when the mode is' train '. The default value is true. The underlying implementation adopts [paddlex.det.transforms.RandomHorizontalFlip](#randomhorizontalflip)

### Add data enhancement methods
```python
ComposedYOLOv3Transforms.add_augmenters(augmenters)
```
> Parameter
>
> * **augmenters**(list): List of data enhancement methods

#### Example
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.ComposedYOLOv3Transforms(mode='train', shape=[480, 480])
eval_transforms = transforms.ComposedYOLOv3Transforms(mode='eval', shape=[480, 480])

# Add data enhancement
import imgaug.augmenters as iaa
train_transforms.add_augmenters([
			iaa.blur.GaussianBlur(sigma=(0.0, 3.0))
])
```
The code above is equivalent to
```
import paddlex as pdx
from paddlex.det import transforms
train_transforms = transforms.Composed([
		iaa.blur.GaussianBlur(sigma=(0.0, 3.0)),
		# The above two are additional data enhancement methods that are added through add_aummenters
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
