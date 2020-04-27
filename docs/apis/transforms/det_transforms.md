# 检测-paddlex.det.transforms

对目标检测任务的数据进行操作。可以利用[Compose](#compose)类将图像预处理/增强操作进行组合。

## Compose类
```python
paddlex.det.transforms.Compose(transforms)
```

根据数据预处理/增强算子对输入数据进行操作。[使用示例](http://gitlab.baidu.com/Paddle/PaddleX/blob/develop/tutorials/train/detection/yolov3_mobilenetv1.py#L13)

### 参数
* **transforms** (list): 数据预处理/数据增强列表。

## ResizeByShort类
```python
paddlex.det.transforms.ResizeByShort(short_size=800, max_size=1333)
```

根据图像的短边调整图像大小（resize）。  
1. 获取图像的长边和短边长度。  
2. 根据短边与short_size的比例，计算长边的目标长度，此时高、宽的resize比例为short_size/原图短边长度。  
3. 如果max_size>0，调整resize比例：
   如果长边的目标长度>max_size，则高、宽的resize比例为max_size/原图长边长度。
4. 根据调整大小的比例对图像进行resize。

### 参数
* **short_size** (int): 短边目标长度。默认为800。
* **max_size** (int): 长边目标长度的最大限制。默认为1333。

## Padding类
```python
paddlex.det.transforms.Padding(coarsest_stride=1)
```

将图像的长和宽padding至coarsest_stride的倍数。如输入图像为[300, 640], `coarest_stride`为32，则由于300不为32的倍数，因此在图像最右和最下使用0值进行padding，最终输出图像为[320, 640]
1. 如果coarsest_stride为1则直接返回。
2. 计算宽和高与最邻近的coarest_stride倍数差值
3. 根据计算得到的差值，在图像最右和最下进行padding

### 参数
* **coarsest_stride** (int): 填充后的图像长、宽为该参数的倍数，默认为1。

## Resize类
```python
paddlex.det.transforms.Resize(target_size=608, interp='LINEAR')
```

调整图像大小（resize）。  
* 当目标大小（target_size）类型为int时，根据插值方式，将图像resize为[target_size, target_size]。  
* 当目标大小（target_size）类型为list或tuple时，根据插值方式，将图像resize为target_size。  
【注意】当插值方式为“RANDOM”时，则随机选取一种插值方式进行resize，作为模型训练时的数据增强操作。

### 参数
* **target_size** (int/list/tuple): 短边目标长度。默认为608。
* **interp** (str): resize的插值方式，与opencv的插值方式对应，取值范围为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为"LINEAR"。

## RandomHorizontalFlip类
```python
paddlex.det.transforms.RandomHorizontalFlip(prob=0.5)
```

以一定的概率对图像进行随机水平翻转，模型训练时的数据增强操作。

### 参数
* **prob** (float): 随机水平翻转的概率。默认为0.5。

## Normalize类
```python
paddlex.det.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

对图像进行标准化。  
1. 归一化图像到到区间[0.0, 1.0]。  
2. 对图像进行减均值除以标准差操作。

### 参数
* **mean** (list): 图像数据集的均值。默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像数据集的标准差。默认为[0.229, 0.224, 0.225]。

## RandomDistort类
```python
paddlex.det.transforms.RandomDistort(brightness_range=0.5, brightness_prob=0.5, contrast_range=0.5, contrast_prob=0.5, saturation_range=0.5, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```

以一定的概率对图像进行随机像素内容变换，模型训练时的数据增强操作。  
1. 对变换的操作顺序进行随机化操作。
2. 按照1中的顺序以一定的概率对图像在范围[-range, range]内进行随机像素内容变换。

### 参数
* **brightness_range** (float): 明亮度因子的范围。默认为0.5。
* **brightness_prob** (float): 随机调整明亮度的概率。默认为0.5。
* **contrast_range** (float): 对比度因子的范围。默认为0.5。
* **contrast_prob** (float): 随机调整对比度的概率。默认为0.5。
* **saturation_range** (float): 饱和度因子的范围。默认为0.5。
* **saturation_prob** (float): 随机调整饱和度的概率。默认为0.5。
* **hue_range** (int): 色调因子的范围。默认为18。
* **hue_prob** (float): 随机调整色调的概率。默认为0.5。

## MixupImage类
```python
paddlex.det.transforms.MixupImage(alpha=1.5, beta=1.5, mixup_epoch=-1)
```

对图像进行mixup操作，模型训练时的数据增强操作，目前仅YOLOv3模型支持该transform。  
当label_info中不存在mixup字段时，直接返回，否则进行下述操作：
1. 从随机beta分布中抽取出随机因子factor。  
2. 根据不同情况进行处理：
    * 当factor>=1.0时，去除label_info中的mixup字段，直接返回。  
    * 当factor<=0.0时，直接返回label_info中的mixup字段，并在label_info中去除该字段。  
    * 其余情况，执行下述操作：  
    （1）原图像乘以factor，mixup图像乘以(1-factor)，叠加2个结果。  
    （2）拼接原图像标注框和mixup图像标注框。  
    （3）拼接原图像标注框类别和mixup图像标注框类别。  
    （4）原图像标注框混合得分乘以factor，mixup图像标注框混合得分乘以(1-factor)，叠加2个结果。
3. 更新im_info中的augment_shape信息。

### 参数
* **alpha** (float): 随机beta分布的下限。默认为1.5。
* **beta** (float): 随机beta分布的上限。默认为1.5。
* **mixup_epoch** (int): 在前mixup_epoch轮使用mixup增强操作；当该参数为-1时，该策略不会生效。默认为-1。

## RandomExpand类
```python
paddlex.det.transforms.RandomExpand(max_ratio=4., prob=0.5, mean=[127.5, 127.5, 127.5])
```

随机扩张图像，模型训练时的数据增强操作，模型训练时的数据增强操作。  
1. 随机选取扩张比例（扩张比例大于1时才进行扩张）。  
2. 计算扩张后图像大小。  
3. 初始化像素值为数据集均值的图像，并将原图像随机粘贴于该图像上。  
4. 根据原图像粘贴位置换算出扩张后真实标注框的位置坐标。

### 参数
* **max_ratio** (float): 图像扩张的最大比例。默认为4.0。
* **prob** (float): 随机扩张的概率。默认为0.5。
* **mean** (list): 图像数据集的均值（0-255）。默认为[127.5, 127.5, 127.5]。

## RandomCrop类
```python
paddlex.det.transforms.RandomCrop(batch_sampler=None, satisfy_all=False, avoid_no_bbox=True)
```

随机裁剪图像，模型训练时的数据增强操作。  
1. 根据batch_sampler计算获取裁剪候选区域的位置。  
    (1) 根据min scale、max scale、min aspect ratio、max aspect ratio计算随机剪裁的高、宽。  
    (2) 根据随机剪裁的高、宽随机选取剪裁的起始点。  
    (3) 筛选出裁剪候选区域：  
    * 当satisfy_all为True时，需所有真实标注框与裁剪候选区域的重叠度满足需求时，该裁剪候选区域才可保留。  
    * 当satisfy_all为False时，当有一个真实标注框与裁剪候选区域的重叠度满足需求时，该裁剪候选区域就可保留。  
2. 遍历所有裁剪候选区域：  
    (1) 若真实标注框与候选裁剪区域不重叠，或其中心点不在候选裁剪区域，则将该真实标注框去除。  
    (2) 计算相对于该候选裁剪区域，真实标注框的位置，并筛选出对应的类别、混合得分。  
    (3) 若avoid_no_bbox为False，返回当前裁剪后的信息即可；反之，要找到一个裁剪区域中真实标注框个数不为0的区域，才返回裁剪后的信息。

### 参数
* **batch_sampler** (list): 随机裁剪参数的多种组合，每种组合包含8个值，如下：
    - max sample (int)：满足当前组合的裁剪区域的个数上限。
    - max trial (int): 查找满足当前组合的次数。
    - min scale (float): 裁剪面积相对原面积，每条边缩短比例的最小限制。
    - max scale (float): 裁剪面积相对原面积，每条边缩短比例的最大限制。
    - min aspect ratio (float): 裁剪后短边缩放比例的最小限制。
    - max aspect ratio (float): 裁剪后短边缩放比例的最大限制。
    - min overlap (float): 真实标注框与裁剪图像重叠面积的最小限制。
    - max overlap (float): 真实标注框与裁剪图像重叠面积的最大限制。

    默认值为None，当为None时采用如下设置：

    [[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],  
     [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 1.0],  
     [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 1.0],  
     [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 1.0],  
     [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 1.0],  
     [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 1.0],  
     [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]
* **satisfy_all** (bool): 是否需要所有标注框满足条件，裁剪候选区域才保留。默认为False。
* **avoid_no_bbox** (bool)： 是否对裁剪图像不存在标注框的图像进行保留。默认为True。
