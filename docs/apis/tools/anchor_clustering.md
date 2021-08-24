# YOLO系列模型anchor聚类

YOLO系列模型均支持自定义anchor，我们提供的默认配置为在MS COCO检测数据集上聚类生成的anchor。用户可以使用在自定义数据集上聚类生成的anchor以提升模型在特定数据集上的的精度。

## YOLOAnchorCluster

```python
class paddlex.tools.YOLOAnchorCluster(num_anchors, dataset, image_size, cache, cache_path=None, iters=300, gen_iters=1000, thresh=0.25)
```
分析数据集中所有图像的标签，聚类生成YOLO系列检测模型指定格式的anchor，返回结果按照由小到大排列。

> **注解**
>
> 自定义YOLO系列模型的`anchor`需要同时指定`anchor_masks`参数。`anchor_masks`参数为一个二维的列表，其长度等于模型backbone获取到的特征图数量（对于PPYOLO的MobileNetV3和ResNet18_vd，特征图数量为2，其余情况为3）。列表中的每一个元素也为列表，代表对应特征图上所检测的anchor编号。
> 以PPYOLO网络的默认参数`anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]`，`anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]`为例，代表在第一个特征图上检测尺度为`[116, 90], [156, 198], [373, 326]`的目标，在第二个特征图上检测尺度为`[30, 61], [62, 45], [59, 119]`的目标，以此类推。

> **参数**
>
* **num_anchors** (int): 生成anchor的数量。PPYOLO，当backbone网络为MobileNetV3或ResNet18_vd时通常设置为6，其余情况通常设置为9。对于PPYOLOv2、PPYOLOTiny、YOLOv3，通常设置为9。
* **dataset** (paddlex.dataset)：用于聚类生成anchor的检测数据集，支持`VOCDetection`和`CocoDetection`格式。
* **image_size** (List[int] or int)：训练时网络输入的尺寸。如果为list，长度须为2，分别代表高和宽；如果为int，代表输入尺寸高和宽相同。
* **cache** (bool): 是否使用缓存。聚类生成anchor需要遍历数据集统计所有真值框的尺寸以及所有图片的尺寸，较为耗时。如果为True，会将真值框尺寸信息以及图片尺寸信息保存至`cache_path`路径下，若路径下已存缓存文件，则加载该缓存。如果为False，则不会保存或加载。默认为True。
* **cache_path** (None or str)：真值框尺寸信息以及图片尺寸信息缓存路径。 如果为None，则使用数据集所在的路径`data_dir`。默认为None。
* **iters** (int)：K-Means聚类算法迭代次数。
* **gen_iters** (int)：基因演算法迭代次数。
* **thresh** (float)：anchor尺寸与真值框尺寸之间比例的阈值。

**代码示例**
```python
import paddlex as pdx
from paddlex import transforms as T

# 下载和解压昆虫检测数据集
dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(dataset, path='./')

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
train_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/train_list.txt',
    label_list='insect_det/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/val_list.txt',
    label_list='insect_det/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

# 在训练集上聚类生成9个anchor
cluster = pdx.tools.YOLOAnchorCluster(num_anchors=9,
                                      dataset=train_dataset,
                                      image_size=608)
anchors = cluster()
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/train/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.det.PPYOLO(num_classes=num_classes,
                       backbone='ResNet50_vd_dcn',
                       anchors=anchors,
                       anchor_masks=anchor_masks)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/detection.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/parameters.md
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    pretrain_weights='COCO',
    learning_rate=0.005 / 12,
    warmup_steps=500,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    lr_decay_epochs=[85, 135],
    save_dir='output/ppyolo_r50vd_dcn',
    use_vdl=True)
```
