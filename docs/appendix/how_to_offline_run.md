# 无联网模型训练

PaddleX在模型训练时，存在以下两种情况需要进行联网下载
> 1.训练模型时，用户没有配置自定义的预训练模型权重`pretrain_weights`，此时PaddleX会自动联网下载在标准数据集上的预训练模型；
> 2.模型裁剪训练时，用户没有配置自定义的参数敏感度信息文件`sensitivities_file`，并将`sensitivities_file`配置成了'DEFAULT'字符串，此时PaddleX会自动联网下载模型在标准数据集上计算得到的参数敏感度信息文件。


用户可以通过本文最末的代码先下载好所有的预训练模型到指定的目录（在代码中我们下载到了`/home/work/paddlex_pretrain`目录)

在训练模型时，需要配置paddlex全局预训练模型路径，将此路径指定到存放了所有预训练模型的路径即可，如下示例代码
```
import paddlex as pdx
# 在import paddlex指定全局的预训练模型路径
# 模型训练时会跳过下载的过程，使用该目录下载好的模型
pdx.pretrain_dir = '/home/work/paddlex_pretrain'
```
按上方式配置后，之后即可进行无联网模型训练


### 下载所有预训练模型代码

> 所有预训练模型下载解压后约为7.5G
```
from paddlex.cv.models.utils.pretrain_weights import image_pretrain
from paddlex.cv.models.utils.pretrain_weights import coco_pretrain
import paddlehub as hub

save_dir = '/home/work/paddlex_pretrain'
for name, url in image_pretrain.items():
    hub.download(name, save_dir)
for name, url in coco_pretrain.items():
    hub.download(name, save_dir)
```
