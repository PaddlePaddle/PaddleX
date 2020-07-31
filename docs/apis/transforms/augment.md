# 数据增强与imgaug支持

数据增强操作可用于在模型训练时，增加训练样本的多样性，从而提升模型的泛化能力。

## PaddleX内置增强操作

PaddleX对于图像分类、目标检测、实例分割和语义分割内置了部分常见的数据增强操作，如下表所示，

| 任务类型 | 增强方法     |
| :------- | :------------|
| 图像分类 | [RandomCrop](cls_transforms.md#randomcrop)、[RandomHorizontalFlip](cls_transforms.md#randomhorizontalflip)、[RandomVerticalFlip](cls_transforms.md#randomverticalflip)、 <br> [RandomRotate](cls_transforms.md#randomrotate)、 [RandomDistort](cls_transforms.md#randomdistort) |
|目标检测<br>实例分割| [RandomHorizontalFlip](det_transforms.md#randomhorizontalflip)、[RandomDistort](det_transforms.md#randomdistort)、[RandomCrop](det_transforms.md#randomcrop)、<br> [MixupImage](det_transforms.md#mixupimage)(仅支持YOLOv3模型)、[RandomExpand](det_transforms.md#randomexpand) |
|语义分割  | [RandomHorizontalFlip](seg_transforms.md#randomhorizontalflip)、[RandomVerticalFlip](seg_transforms.md#randomverticalflip)、[ResizeRangeScaling](seg_transforms.md#resizerangescaling)、<br> [ResizeStepScaling](seg_transforms.md#resizerangescaling)、[RandomPaddingCrop](seg_transforms.md#randompaddingcrop)、 [RandomBlur](seg_transforms.md#randomblur)、<br> [RandomRotate](seg_transforms.md#randomrotate)、[RandomScaleAspect](seg_transforms.md#randomscaleaspect)、[RandomDistort](seg_transforms.md#randomdistort) |

## imgaug增强库的支持

PaddleX目前已适配imgaug图像增强库，用户可以直接在PaddleX构造`transforms`时，调用imgaug的方法, 如下示例
```
import paddlex as pdx
from paddlex.cls import transforms
import imgaug.augmenters as iaa
train_transforms = transforms.Compose([
    # 随机在[0.0 3.0]中选值对图像进行模糊
    iaa.blur.GaussianBlur(sigma=(0.0, 3.0)),
    transforms.RandomCrop(crop_size=224),
    transforms.Normalize()
])
```
除了上述用法，`Compose`接口中也支持imgaug的`Someof`、`Sometimes`、`Sequential`、`Oneof`等操作，开发者可以通过这些方法随意组合出增强流程。由于imgaug对于标注信息(目标检测框和实例分割mask)与PaddleX模型训练逻辑有部分差异，**目前在检测和分割中，只支持pixel-level的增强方法,（即在增强时，不对图像的大小和方向做改变） 其它方法仍在适配中**，详情可见下表，

| 增强方法 | 图像分类 | 目标检测<br> 实例分割 | 语义分割 | 备注 |
| :------  | :------- | :-------------------- | :------- | :--- |
| [imgaug.augmenters.arithmetic](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html) |√ |√ |√ | Cutout, Dropout, JpegCompression等|
| [imgaug.augmenters.artistic](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_artistic.html) |√ |√ |√ | 图像卡通化|
| [imgaug.augmenters.blur](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html) |√ |√ |√ | GaussianBlur, AverageBlur等|
| [imgaug.augmenters.collections](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_collections.html) |√ | | |提供了RandAugment方法 |
| [imgaug.augmenters.color](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_color.html) |√ |√ |√ | Brightness, Hue等色调的增强方法|
| [imgaug.augmenters.contrast](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_contrast.html) |√ |√ |√ | 多种对比度增强方式|
| [imgaug.augmenters.convolutional](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_convolutional.html) |√ |√ |√ | 应用卷积kernel到图像 |
| [imgaug.augmenters.edges](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_edges.html) |√ |√ |√ | 图像边缘化等方法|
| [imgaug.augmenters.flip](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_flip.html) |√ | | | Fliplr和Flipud翻转方法|
| [imgaug.augmenters.geometric](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html) |√ | | | Affine、Rotate等增强方法|
| [imgaug.augmenters.imgcorruptlike](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_imgcorruptlike.html) |√ |√ |√ | GaussianNoise等图像噪声增强方法|
| [imgaug.augmenters.pillike](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_pillike.html) |√ | | | |
| [imgaug.augmenters.pooling](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_pooling.html) |√ | | |应用pooling操作到图像 |
| [imgaug.augmenters.segmentation](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_segmentation.html) |√ | | | 应用分割方法到图像|
| [imgaug.augmenters.size](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_size.html) |√ | | | Reisze、Crop、Pad等操作|
| [imgaug.augmenters.weather](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_weather.html) |√ |√ |√ | 多种模拟天气等增强方法|

需要注意的是，imgaug的基础方法中，如`imgaug.augmenters.blur`仅为图像处理操作，并无概率设置，而在CV模型训练中，增强操作往往是以一定概率应用在样本上，因此我们可以通过imgaug的`Someof`、`Sometimes`、`Sequential`、`Oneof`等操作来组合实现，如下代码所示，
> - `Someof` 执行定义增强方法列表中的部分方法
> - `Sometimes` 以一定概率执行定义的增强方法列表
> - `Sequential` 按顺序执行定义的增强方法列表
```
image imgaug.augmenters as iaa
from paddlex.cls import transforms
# 以0.6的概率对图像样本进行模糊
img_augmenters = iaa.Sometimes(0.6, [
    iaa.blur.GaussianBlur(sigma=(0.0, 3.0))
])
train_transforms = transforms.Compose([
    img_augmenters,
    transforms.RandomCrop(crop_size=224),
    transforms.Normalize()
])
```
