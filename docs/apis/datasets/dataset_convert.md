# 数据集转换
## labelme2voc
```python
pdx.tools.labelme2voc(image_dir, json_dir, dataset_save_dir)
```
将LabelMe标注的数据集转换为VOC数据集。

> **参数**
> > * **image_dir** (str): 图像文件存放的路径。
> > * **json_dir** (str): 与每张图像对应的json文件的存放路径。
> > * **dataset_save_dir** (str): 转换后数据集存放路径。

## 其它数据集转换
### easydata2imagenet
```python
pdx.tools.easydata2imagenet(image_dir, json_dir, dataset_save_dir)
```
### easydata2voc
```python
pdx.tools.easydata2voc(image_dir, json_dir, dataset_save_dir)
```
### easydata2coco
```python
pdx.tools.easydata2coco(image_dir, json_dir, dataset_save_dir)
```
### easydata2seg
```python
pdx.tools.easydata2seg(image_dir, json_dir, dataset_save_dir)
```
### labelme2coco
```python
pdx.tools.labelme2coco(image_dir, json_dir, dataset_save_dir)
```
### labelme2seg
```python
pdx.tools.labelme2seg(image_dir, json_dir, dataset_save_dir)
```
### jingling2seg
```python
pdx.tools.jingling2seg(image_dir, json_dir, dataset_save_dir)
```

