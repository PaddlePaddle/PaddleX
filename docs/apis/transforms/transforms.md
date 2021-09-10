# paddlex.transforms

对图像分类、语义分割、目标检测、实例分割任务的数据进行操作。可以利用[Compose](#compose)类将图像预处理/增强操作进行组合。

## 目录

* [Compose](#1)
* [Normalize](#2)
* [Resize](#3)
* [RandomResize](#4)
* [ResizeByShort](#5)
* [RandomResizeByShort](#6)
* [ResizeByLong](#7)
* [RandomHorizontalFlip](#8)
* [RandomVerticalFlip](#9)
* [CenterCrop](#10)
* [RandomCrop](#11)
* [RandomScaleAspect](#12)
* [RandomExpand](#13)
* [Padding](#14)
* [MixupImage](#15)
* [RandomDistort](#16)
* [RandomBlur](#17)

## <h2 id="1">Compose</h2>
```python
paddlex.transforms.Compose(transforms)
```

根据数据预处理/增强算子对输入数据进行操作。  [使用示例](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/tutorials/train/image_classification/mobilenetv3_small.py#L10)

> **参数**
> * **transforms** (List[paddlex.transforms.Transform]): 数据预处理/数据增强列表。

## <h2 id="2">Normalize</h2>
```python
paddlex.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], min_val=[0, 0, 0], max_val=[255., 255., 255.], is_scale=True)
```
对图像进行标准化。

1. 像素值减去min_val
2. 像素值除以(max_val-min_val), 归一化到区间 [0.0, 1.0]。
3. 对图像进行减均值除以标准差操作。

> **参数**
> * **mean** (list): 图像数据集的均值。默认为[0.485, 0.456, 0.406]。长度应与图像通道数量相同。
> * **std** (list): 图像数据集的标准差。默认为[0.229, 0.224, 0.225]。长度应与图像通道数量相同。
> * **min_val** (list): 图像数据集的最小值。默认值[0, 0, 0]。长度应与图像通道数量相同。
> * **max_val** (list): 图像数据集的最大值。默认值[255.0, 255.0, 255.0]。长度应与图像通道数量相同。
> * **is_scale** (bool): 是否将图片像素值除以255。默认为True。

## <h2 id="3">Resize</h2>
```python
paddlex.transfroms.Resize(target_size, interp='LINEAR', keep_ratio=False)
```
调整图像大小。

> **参数**
> * **target_size** (int, List[int] or Tuple[int]): 目标尺寸。如果为int，图像的高和宽共用同一目标尺寸。如果为List[int]或Tuple[int]，长度须为2，分别代表图像的目标高度和目标宽度。
> * **interp** (str): 调整图像尺寸时使用的插值方法。取值范围为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为'LINEAR'。
> * **keep_ratio** (bool): 是否保留图像的原始高宽比。默认为False。

## <h2 id="4">RandomResize</h2>
```python
paddlex.transfroms.RandomResize(target_sizes, interp='LINEAR')
```
随机选取目标尺寸调整图像大小。

> **参数**
> * **target_sizes** (List[int], List[list or tuple] or Tuple[list or tuple]): 备选目标尺寸列表，每张图像会从中随机选取目标尺寸值。如果为List[int]，图像的高和宽共用同一目标尺寸。如果为List[list or tuple]或Tuple[list or tuple]，每一个元素长度须为2，分别代表图像的目标高度和目标宽度。
> * **interp** (str): 调整图像尺寸时使用的插值方法。取值范围为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为'LINEAR'。

## <h2 id="5">ResizeByShort</h2>
```python
paddlex.transforms.ResizeByShort(short_size=256, max_size=-1, interp='LINEAR')
```

根据图像的短边调整图像大小。  
1. 获取图像的长边和短边长度。  
2. 根据短边与short_size的比例，计算长边的目标长度，此时高、宽的缩放比例为short_size/原图短边长度。  
3. 如果max_size>0，调整resize比例：
   如果长边的目标长度>max_size，则高、宽的缩放比例为max_size/原图长边长度。
4. 根据高、宽的缩放比例调整图像尺寸。

> **参数**
> * **short_size** (int): 调整大小后的图像目标短边长度。默认为256。
> * **max_size** (int): 长边目标长度的最大限制。默认为-1。
> * **interp** (str): 调整图像尺寸时使用的插值方法。取值范围为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为'LINEAR'。

## <h2 id="6">RandomResizeByShort</h2>
```python
paddlex.transforms.RandomResizeByShort(short_sizes, max_size=-1, interp='LINEAR')
```
随机选取目标尺寸，根据图像的短边调整图像大小。
> **参数**
> * **short_size** (List[int]): 调整大小后的图像目标短边长度备选值列表，每张图像会从中随机选取目标尺寸值。
> * **max_size** (int): 长边目标长度的最大限制。默认为-1。
> * **interp** (str): 调整图像尺寸时使用的插值方法。取值范围为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为'LINEAR'。

## <h2 id="7">ResizeByLong</h2>

```python
paddlex.transforms.ResizeByLong(long_size=256, interp='LINEAR')
```
图像长边调整到固定值，短边按比例进行缩放。
> **参数**
> * **long_size** (int): 调整大小后图像的长边长度。
> * **interp** (str): 调整图像尺寸时使用的插值方法。取值范围为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为'LINEAR'。

## <h2 id="8">RandomHorizontalFlip</h2>
```python
paddlex.cls.transforms.RandomHorizontalFlip(prob=0.5)
```
以一定的概率对图像进行随机水平翻转，模型训练时的数据增强操作。
> **参数**
> * **prob** (float): 随机水平翻转的概率。默认为0.5。

## <h2 id="9">RandomVerticalFlip</h2>
```python
paddlex.cls.transforms.RandomVerticalFlip(prob=0.5)
```
以一定的概率对图像进行随机垂直翻转，模型训练时的数据增强操作。
> **参数**
> * **prob** (float): 随机垂直翻转的概率。默认为0.5。

## <h2 id="10">CenterCrop</h2>
```python
paddlex.transforms.CenterCrop(crop_size=224)
```
以图像中心点扩散裁剪长宽为目标尺寸的正方形  
1. 计算剪裁的起始点。  
2. 剪裁图像。
> **参数**
> * **crop_size** (int): 裁剪的目标边长。默认为224。

## <h2 id="11">RandomCrop</h2>
```python
paddlex.transforms.RandomCrop(crop_size=None, aspect_ratio=[.5, 2.], thresholds=[.0, .1, .3, .5, .7, .9], scaling=[.3, 1.], num_attempts=50, allow_no_crop=True, cover_all_box=False)
```
随机裁剪图像，模型训练时的数据增强操作。

如果为检测或实例分割任务，操作步骤如下：
1. 若`allow_no_crop`为True，则在`thresholds`加入’no_crop’。
2. 随机打乱`thresholds`。
3. 遍历`thresholds`中各元素（thresh）：
   1. 如果当前thresh为’no_crop’，则返回原始图像和标注信息。
   2. 随机取出`aspect_ratio`和`scaling`中的值并由此计算出候选裁剪区域的高、宽、起始点。
   3. 计算真实标注框与候选裁剪区域IoU，若全部真实标注框的IoU都小于thresh，则继续第3步。
   4. 如果`cover_all_box`为True且存在真实标注框的IoU小于thresh，则继续第3步。
   5. 筛选出位于候选裁剪区域内的真实标注框，若有效框的个数为0，则继续第3步，否则进行第4步。
4. 换算有效真值标注框相对候选裁剪区域的位置坐标。
5. 换算有效分割区域相对候选裁剪区域的位置坐标。

对于其他任务，随机取出`aspect_ratio`和`scaling`中的值并由此计算出候选裁剪区域的高、宽、起始点对图像进行裁切。
> **参数**
> * **crop_size** (int, List[int], Tuple[int] or None): 随机裁剪后重新调整的目标尺寸。如果为None，裁剪后不会对图像尺寸进行调整。默认为None。
> * **aspect_ratio** (List(float)): 裁剪后高宽比的取值范围，以[min, max]形式表示。默认值为[.5, 2.]。
> * **thresholds** (List(float)): 判断裁剪候选区域是否有效所需的IoU阈值取值列表。该参数仅在检测和实例分割任务中有效。默认值为[.0, .1, .3, .5, .7, .9]。
> * **scaling** (List(float)): 裁剪面积相对原面积的取值范围，以[min, max]形式表示。默认值为[.3, 1.]。
> * **num_attempts** (int): 在放弃寻找有效裁剪区域前尝试的次数。默认值为50。
> * **allow_no_crop** (bool): 是否允许未进行裁剪。该参数仅在检测和实例分割任务中有效。默认值为True。
> * **cover_all_box** (bool): 是否要求所有的真实标注框都必须在裁剪区域内。该参数仅在检测和实例分割任务中有效。默认值为False。

## <h2 id="12">RandomScaleAspect</h2>
```python
paddlex.transforms.RandomScaleAspect(min_scale=0.5, aspect_ratio=0.33)
```
按照一定的面积比和高宽比对图像进行裁剪，并调整回图像原始尺寸。
> **参数**
> * **min_scale**  (float)：裁取图像占原始图像的面积比，取值[0，1]，为0时则返回原图。默认为0.5。
> * **aspect_ratio** (float): 裁取图像的高宽比范围，非负值，为0时返回原图。默认为0.33。

## <h2 id="13">RandomExpand</h2>
```python
paddlex.transforms.RandomExpand(upper_ratio=4., prob=.5, im_padding_value=(127.5, 127.5, 127.5), label_padding_value=255)
```
随机扩张图像，模型训练时的数据增强操作。
1. 随机选取扩张比例（扩张比例大于1时才进行扩张）。
2. 计算扩张后图像大小。
3. 初始化像素值为输入填充值的图像，并将原图像随机粘贴于该图像上。

> **参数**
> * **upper_ratio** (float): 图像扩张的最大比例。默认为4.0。
> * **prob** (float): 随机扩张的概率。默认为0.5。
> * **im_padding_value** (list): 扩张图像的初始填充值（0-255）。默认为[127.5, 127.5, 127.5]。
> * **label_padding_value** (int): 扩张标注的填充值。该参数仅在分割任务中有效。默认为255。

【注意】该数据增强必须在数据增强Resize、ResizeByShort之前使用。

## <h2 id="14">Padding</h2>
```python
paddlex.transforms.Padding(target_size=None, pad_mode=0, offsets=None, im_padding_value=(127.5, 127.5, 127.5), label_padding_value=255, size_divisor=32)
```
使用指定像素值对图像进行填充。
> **参数**
> * **target_size** (int, List[int], Tuple[int] or None): 填充后的目标尺寸。如果为int，填充后的高和宽均为`target_size`。如果为List[int]或Tuple[int]，长度须为2，分别表示填充后图像的高度和宽度。如果为None，则将图像的高和宽均调整到最近的`size_divisor`的倍数。默认为None。
> * **pad_model** (int): 填充使用的模式，取值范围为[-1, 0, 1, 2]。如果为-1，根据指定的`offsets`进行填充；如果为0，仅向原始图像右下方进行填充；如果为1，向原始图像四个方向均匀填充；如果为2，仅向原始图像左上方进行填充。默认为0。
> * **offsets** (List[int] or None): 填充后原始图像左上角坐标值，长度须为2，分别代表左侧填充宽度和上方填充宽度（右侧填充宽度和下方填充宽度会根据指定的`target_size`进行计算）。该参数仅在pad_mode为-1是有效。
> * **im_padding_value** (List[float]): 图像填充的值。长度须与图像通道数量相同，代表每个通道的填充值。默认为[127.5, 127.5, 127.5]。
> * **label_padding_value** (int): 标注图像padding的值。默认值为255（仅在训练时需要设定该参数）。该参数仅在分割任务中有效。
> * **size_divisor** (int): 如果`target_size`为None，将图像的高和宽均调整到最近的`size_divisor`的倍数。

## <h2 id="15">MixupImage</h2>
```python
paddlex.transforms.MixupImage(alpha=1.5, beta=1.5, mixup_epoch=-1)
```
对图像进行 [mixup](https://arxiv.org/abs/1710.09412) 操作，模型训练时的数据增强操作。

操作步骤如下：
1. 从数据集中随机选取一张图像。
2. 从随机beta分布中抽取出随机因子factor。
3. 根据不同情况进行处理：
   * 当factor>=1.0时，返回当前图像。
   * 当factor<=0.0时，返回随机选取到的图像。
   * 其余情况，执行下述操作：  
     （1）原图像乘以factor，mixup图像乘以(1-factor)，叠加2个结果。  
     （2）拼接原图像标注框和mixup图像标注框。  
     （3）拼接原图像标注框类别和mixup图像标注框类别。  
     （4）原图像标注框混合得分乘以factor，mixup图像标注框混合得分乘以(1-factor)，叠加2个结果。

> **参数**
> * **alpha** (float): 随机beta分布的下限。默认为1.5。
> * **beta** (float): 随机beta分布的上限。默认为1.5。
> * **mixup_epoch** (int): 在前mixup_epoch轮使用mixup增强操作；当该参数为-1时，该策略会在整个训练过程生效。默认为-1。

## <h2 id="16">RandomDistort</h2>
```python
paddlex.transforms.RandomDistort(brightness_range=0.5, brightness_prob=0.5, contrast_range=0.5, contrast_prob=0.5, saturation_range=0.5, saturation_prob=0.5, hue_range=18, hue_prob=0.5, random_apply=True, count=4, shuffle_channel=False)```
```
以一定的概率对图像进行随机像素内容变换，可包括亮度、对比度、饱和度、色相角度、通道顺序的调整，模型训练时的数据增强操作。

【注意】如果输入是uint8/uint16的RGB图像，该数据增强必须在数据增强Normalize之前使用。

> **参数**
> * **brightness_range** (float): 明亮度的缩放系数范围。从[1-`brightness_range`, 1+`brightness_range`]中随机取值作为明亮度缩放因子`scale`，按照公式`image = image * scale`调整图像明亮度。默认值为0.9。
> * **brightness_prob** (float): 随机调整明亮度的概率。默认为0.5。
> * **contrast_range** (float): 对比度的缩放系数范围。从[1-`contrast_range`, 1+`contrast_range`]中随机取值作为对比度缩放因子`scale`，按照公式`image = image * scale + (image_mean + 0.5) * (1 - scale)`调整图像对比度。默认为0.9。
> * **contrast_prob** (float): 随机调整对比度的概率。默认为0.5。
> * **saturation_range** (float): 饱和度的缩放系数范围。从[1-`saturation_range`, 1+`saturation_range`]中随机取值作为饱和度缩放因子`scale`，按照公式`image = gray * (1 - scale) + image * scale`，其中`gray = R * 299/1000 + G * 587/1000+ B * 114/1000`。默认为0.9。
> * **saturation_prob** (float): 随机调整饱和度的概率。默认为0.5。
> * **hue_range** (int): 调整色相角度的差值取值范围。从[-`hue_range`, `hue_range`]中随机取值作为色相角度调整差值`delta`，按照公式`hue = hue + delta`调整色相角度 。默认为18，取值范围[0, 360]。
> * **hue_prob** (float): 随机调整色调的概率。默认为0.5。
> * **random_apply** (bool): 是否采用随机顺序进行系列变换。如果为True，从亮度、对比度、饱和度、色相角度中随机选取`count`个，并按照随机顺序进行调整。如果为False，按照亮度、对比度、饱和度、色相角度或按照亮度、饱和度、色相角度、对比度的顺序依次进行调整，两种顺序的概率均为0.5。
> * **count** (int): 当`random_apply`为True时，选取的像素变换种类个数。默认为4。
> * **shuffle_channel** (bool): 是否随机调整通道顺序。默认为False。

## <h2 id="17">RandomBlur</h2>
```python
paddlex.transforms.RandomBlur(prob=0.1)
```
以一定的概率对图像进行高斯模糊，模型训练时的数据增强操作。
> **参数**
> * **prob** (float): 图像模糊概率。默认为0.1。
