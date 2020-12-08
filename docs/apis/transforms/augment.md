# Data augmenter and imgaug support

Data augmenter operations can be used to increase the diversity of the training samples during model training, thereby improving the generalization of the models.

## PaddleX Built-in Augmentater Operations

PaddleX has some common data augmenter operations built in for image classification, object detection, instance segmentation, and semantic segmentation. See the following table.

| Task type | Augmenter method |
| :------- | :------------|
| Image Classification | [RandomCrop](cls_transforms.html#randomcrop), [RandomHorizontalFlip](cls_transforms.html#randomhorizontalflip), [RandomVerticalFlip](cls_transforms.html#randomverticalflip),<br> [RandomRotate](cls_transforms.html#randomratate), [RandomDistort](cls_transforms.html#randomdistort) |
| Object detection instance segmentation<br> | [RandomHorizontalFlip](det_transforms.html#randomhorizontalflip), [RandomDistort](det_transforms.html#randomdistort), [RandomCrop](det_transforms.html#randomcrop).<br> [MixupImage](det_transforms.html#mixupimage) (YOLOv3 model only), [RandomExpand](det_transforms.html#randomexpand) |
| Semantic Segmentation | [RandomHorizontalFlip](seg_transforms.html#randomhorizontalflip), [RandomVerticalFlip](seg_transforms.html#randomverticalflip), [ResizeRangeScaling](seg_transforms.html#resizerangescaling).<br> [ResizeStepScaling](seg_transforms.html#resizestepscaling), [RandomPaddingCrop](seg_transforms.html#randompaddingcrop), [RandomBlur](seg_transforms.html#randomblur).<br> [RandomRotate](seg_transforms.html#randomrotate), [RandomScaleAspect](seg_transforms.html#randomscaleaspect), [RandomDistort](seg_transforms.html#randomdistort) |

## imgaug augmenter library support

Currently, PaddleX is adapted to the image augment library imgaug. You can directly construct the `transforms` in PaddleX by calling imgaug. The methods are as follows:
```
import paddlex as pdx from paddlex.cls import transforms import imgaug.augmenters as iaa train_transforms = transforms. Compose([ # Randomly blur the image by selecting the value in [0.0 3.0]. iaa.blur. GaussianBlur(sigma=(0.0, 3.0)), transforms. RandomCrop(crop_size=224), transforms. Normalize() ])
```
In addition to the above usage, the `Compose` interface also supports imgaug's `Someof`, `Sometimes`, `Sequential`, `Oneof`, and so on. Developers are free to combine these methods to create an augmenter process. Since imgaug's training logic for labeling information (object detection frame and instance segmentation mask) is different from that in PaddleX models, only the pixel-level augmenter method is supported in detection and segmentation, (that is, the size and orientation of the image are not changed during augmentation).** **

| Augmenter method | Image Classification | Object detection instance segmentation<br> | Semantic Segmentation | Note |
| :------  | :------- | :-------------------- | :------- | :--- |
| [imgaug.augmenters.arithmetic ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html) | √ | √ | √ | Cutout, Dropout, JpegCompression, etc. |
| [imgaug.augmenters.artistic ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_artistic.html) | √ | √ | √ | Cartoonish images |
| [imgaug.augmenters.blur ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html) | √ | √ | √ | GaussianBlur, AverageBlur, etc. |
| [imgaug.augmenters.collections ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_collections.html) | √ |  |  | RandAugment method is provided |
| [imgaug.augmenters.color ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_color.html) | √ | √ | √ | Brightness, Hue and other hue augment methods. |
| [imgaug.augmenters.contrast ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_contrast.html) | √ | √ | √ | Multiple contrast augment methods |
| [imgaug.augmenters.convolutional ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_convolutional.html) | √ | √ | √ | Applying convolutional kernel to images |
| [imgaug.augmenters.edges ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_edges.html) | √ | √ | √ | Methods such as image marginalization |
| [imgaug.augmenters.flip ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_flip.html) | √ |  |  | Fliplr and Flipud flip methods |
| [imgaug.augmenters.geometric ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html) | √ |  |  | Affine, Rotate and other augment methods |
| [imgaug.augmenters.imgcorruptlike ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_imgcorruptlike.html) | √ | √ | √ | Image noise augment methods such as GaussianNoise |
| [imgaug.augmenters.pillike ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_pillike.html) | √ |  |  |  |
| [imgaug.augmenters.pooling ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_pooling.html) | √ |  |  | Apply pooling operations to images |
| [imgaug.augmenters.segmentation ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_segmentation.html) | √ |  |  | Applying segmentation methods to images |
| [imgaug.augmenters.size ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_size.html) | √ |  |  | Reisze, Crop, Pad, and other operations |
| [imgaug.augmenters.weather ](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_weather.html) | √ | √ | √ | Multiple weather simulations and other augments |

It should be noted that the basic methods of imgaug, such as `imgaug.augmenters. blur, are only image processing operations, without the settings of probability. In the CV model training, augment operations are often applied to samples with a certain probability. Therefore, you can combine imgaug's `Someof`, `Sometimes`, `Sequential`, and `Oneof` operations to achieve this. See the following codes:`
> - `Someof`: executes some of the methods in the list of defined augmenter methods
> - `Sometimes`: defines a list of augmenter methods executed in certain probability.
> - `Sequential`: executes the defined list of augmeter methods in order.

```
image imgaug.augmenters as iaa from paddlex.cls import transforms # Blurring of image samples with the probability of 0.6 img_augmenters = iaa. Sometimes(0.6, [ iaa.blur. GaussianBlur(sigma=(0.0, 3.0)) ]) train_transforms = transforms. Compose([ img_augmenters, transforms. RandomCrop(crop_size=224), transforms. Normalize() ])
```
