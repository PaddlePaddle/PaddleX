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
For the data processing processes that have been combined in the classification model, developers can directly use ComposedClsTransforms to simplify the process of manually combining transforms. This class already contains two data enhancement methods [RandomCrop](#RandomCrop) and [RandomHorizontalFlip](#RandomHorizontalFlip). You can still add new data enhancement methods through the [add_augmenters function interface](#add_augmenters). 
ComposedClsTransforms includes the following steps:

> Training stage:
> > 1. Crop a subgraph randomly from the image, and resize it to the size of crop_size
> > 2. Flip the output of 1 at random with a probability of 0.5. If random_horizon_flip is False, skip this step
> > 3. The image was normalized
> Validation / Prediction stage:
> > 1. Resize the image so that the minimum edge length is crop_size [0] * 1.14
> > 2. Crop a crop_size image from the center of the image
> > 3. The image was normalized

### Parameters
* **mode** (str): The stage of Transforms, including 'train', 'eval' or 'test'
* **crop_size** (int|list): The image size input into the model, the default is [224, 224]（It has nothing to do with the size of the original image. According to the above steps, the original image will be processed into the size of the graph and input to the model training)
* **mean** (list): Mean value of the image, the default is [0.485, 0.456, 0.406].
* **std** (list): Variance of the image, the default is [0.229, 0.224, 0.225].
* **random_horizontal_flip**(bool): Whether to use random horizontal flip enhancement with a probability of 0,5 is only effective when the model is' train '. The default value is true.The underlying implementation adopts[paddlex.cls.transforms.RandomHorizontalFlip](#randomhorizontalflip)

### Add data enhancement methods
```python
ComposedClsTransforms.add_augmenters(augmenters)
```
> Parameter
>
> * **augmenters**(list): List of data enhancement methods

#### Example
```
import paddlex as pdx
from paddlex.cls import transforms
train_transforms = transforms.ComposedClsTransforms(mode='train', crop_size=[320, 320])
eval_transforms = transforms.ComposedClsTransforms(mode='eval', crop_size=[320, 320])

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
from paddlex.cls import transforms
train_transforms = transforms.Composed([
		transforms.RandomDistort(),
		iaa.blur.GaussianBlur(sigma=(0.0, 3.0)),
		# The above two are additional data enhancement methods that are added through add_aummenters
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
