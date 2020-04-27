# 分类-paddlex.cls.transforms

对图像分类任务的数据进行操作。可以利用[Compose](#compose)类将图像预处理/增强操作进行组合。

## Compose类
```python
paddlex.cls.transforms.Compose(transforms)
```

根据数据预处理/增强算子对输入数据进行操作。  [使用示例](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/classification/mobilenetv2.py#L13)

### 参数
* **transforms** (list): 数据预处理/数据增强列表。


## RandomCrop类
```python
paddlex.cls.transforms.RandomCrop(crop_size=224, lower_scale=0.88, lower_ratio=3. / 4, upper_ratio=4. / 3)
```

对图像进行随机剪裁，模型训练时的数据增强操作。
1. 根据lower_scale、lower_ratio、upper_ratio计算随机剪裁的高、宽。
2. 根据随机剪裁的高、宽随机选取剪裁的起始点。
3. 剪裁图像。
4. 调整剪裁后的图像的大小到crop_size*crop_size。

### 参数
* **crop_size** (int): 随机裁剪后重新调整的目标边长。默认为224。
* **lower_scale** (float): 裁剪面积相对原面积比例的最小限制。默认为0.88。
* **lower_ratio** (float): 宽变换比例的最小限制。默认为3. / 4。
* **upper_ratio** (float): 宽变换比例的最小限制。默认为4. / 3。

## RandomHorizontalFlip类
```python
paddlex.cls.transforms.RandomHorizontalFlip(prob=0.5)
```

以一定的概率对图像进行随机水平翻转，模型训练时的数据增强操作。

### 参数
* **prob** (float): 随机水平翻转的概率。默认为0.5。

## RandomVerticalFlip类
```python
paddlex.cls.transforms.RandomVerticalFlip(prob=0.5)
```

以一定的概率对图像进行随机垂直翻转，模型训练时的数据增强操作。

### 参数
* **prob** (float): 随机垂直翻转的概率。默认为0.5。

## Normalize类
```python
paddlex.cls.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

对图像进行标准化。  
1. 对图像进行归一化到区间[0.0, 1.0]。  
2. 对图像进行减均值除以标准差操作。

### 参数
* **mean** (list): 图像数据集的均值。默认为[0.485, 0.456, 0.406]。
* **std** (list): 图像数据集的标准差。默认为[0.229, 0.224, 0.225]。

## ResizeByShort类
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

## CenterCrop类
```python
paddlex.cls.transforms.CenterCrop(crop_size=224)
```

以图像中心点扩散裁剪长宽为`crop_size`的正方形  
1. 计算剪裁的起始点。  
2. 剪裁图像。

### 参数
* **crop_size** (int): 裁剪的目标边长。默认为224。

## RandomRotate类
```python
paddlex.cls.transforms.RandomRotate(rotate_range=30, prob=0.5)
```

以一定的概率对图像在[-rotate_range, rotaterange]角度范围内进行旋转，模型训练时的数据增强操作。

### 参数
* **rotate_range** (int): 旋转度数的范围。默认为30。
* **prob** (float): 随机旋转的概率。默认为0.5。

## RandomDistort类
```python
paddlex.cls.transforms.RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5, saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5)
```

以一定的概率对图像进行随机像素内容变换，模型训练时的数据增强操作。  
1. 对变换的操作顺序进行随机化操作。
2. 按照1中的顺序以一定的概率对图像在范围[-range, range]内进行随机像素内容变换。

### 参数
* **brightness_range** (float): 明亮度因子的范围。默认为0.9。
* **brightness_prob** (float): 随机调整明亮度的概率。默认为0.5。
* **contrast_range** (float): 对比度因子的范围。默认为0.9。
* **contrast_prob** (float): 随机调整对比度的概率。默认为0.5。
* **saturation_range** (float): 饱和度因子的范围。默认为0.9。
* **saturation_prob** (float): 随机调整饱和度的概率。默认为0.5。
* **hue_range** (int): 色调因子的范围。默认为18。
* **hue_prob** (float): 随机调整色调的概率。默认为0.5。
