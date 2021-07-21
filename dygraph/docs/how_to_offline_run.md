# 无联网模型训练


PaddleX在模型训练时，用户如果没有将`pretrain_weights`设置为自定义的预训练模型路径，PaddleX就会自动联网下载在标准数据集上的预训练模型。而有的机器无法联网，导致模型训练无法进行。为解决该问题，用户可以事先在另一台联网机器上将所有的标准数据集上的预训练模型下载好，然后拷贝至指定机器上即可。


## PaddleX Python API准备预训练模型

> 用户在可联网的机器上，执行如下代码，所有的预训练模型将会下载至指定的`save_dir`（代码示例中为`/home/work/paddlex_pretrain`），

```
import os.path as osp
import paddlex
from paddlex.utils.checkpoint import cityscapes_weights, imagenet_weights, pascalvoc_weights, coco_weights
from paddlex.utils.download import download_and_decompress

save_dir = '/home/work/paddlex_pretrain'

weights_lists = [cityscapes_weights, imagenet_weights, pascalvoc_weights, coco_weights]
for weights in weights_lists:
    for key, value in weights.items():
        new_save_dir = osp.join(save_dir, key)
        download_and_decompress(value, path=new_save_dir)
```

> 之后在使用PaddleX Python API模式进行PaddleX模型训练时，只需要在import paddlex的同时，配置如下参数，模型在训练时便会优先在此目录下寻找已经下载好的预训练模型。
```
import paddlex as pdx
pdx.pretrain_dir = '/home/work/paddlex_pretrain'
```

## PaddleX GUI准备预训练模型

> PaddleX GUI在打开后，用户可自行设定工作空间，假设当前用户设定的工作空间为`D:\PaddleX_Workspace`。为了能在无联网下完成训练，用户需事先下载所有预训练模型文件至`D:\PaddleX_Workspace\pretrain`目录，之后便不再需要联网下载预训练模型。

> 事先下载所有预训练模型需要依赖PaddleX Python API，如果尚未安装PaddleX Python API，请参考文档[PaddleX API开发模式安装](install.md#1-paddlex-api开发模式安装)进行安装。安装完成后，在已联网的机器上运行以下代码，所有的预训练模型将会下载至指定的`save_dir`（代码示例中为`/home/work/paddlex_pretrain`，也可以直接指定到GUI工作空间下的预训练模型文件存储位置（例如`D:\PaddleX_Workspace\pretrain`）），下载完成后将`save_dir`下的所有文件拷贝至GUI工作空间下的预训练模型文件存储位置（例如`D:\PaddleX_Workspace\pretrain`）下。

```
import paddlex
from paddlex.utils.checkpoint import cityscapes_weights, imagenet_weights, pascalvoc_weights, coco_weights
from paddlex.utils.download import download_and_decompress

save_dir = '/home/work/paddlex_pretrain'

weights_lists = [cityscapes_weights, imagenet_weights, pascalvoc_weights, coco_weights]
for weights in weights_lists:
    for key, value in weights.items():
        download_and_decompress(value, path=new_save_dir)
```
