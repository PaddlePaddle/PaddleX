# 数据标注

### 用户可根据任务种类查看标注文档
- [图像分类数据标注](classification.md)
- [目标检测数据标注](object_detection.md)
- [实例分割数据标注](instance_segmentation.md)
- [语义分割数据标注](semantic_segmentation.md)

### 手机拍照图片旋转

当您收集的样本图像来源于手机拍照时，请注意由于手机拍照信息内附带水平垂直方向信息，这可能会使得在标注和训练时出现问题，因此在拍完照后注意根据方向对照片进行处理，使用如下函数即可解决
```python
from PIL import Image, ExifTags
def rotate(im):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(im._getexif().items())
        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        if exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        if exif[orientation] == 8:
            im = im.rotate(90, expand=True)
    except:
        pass

img_file = '1.jpeg'
im = Image.open(img_file)
rotate(im)
im.save('new_1.jpeg')
```
