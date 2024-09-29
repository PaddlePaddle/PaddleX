通用语义分割产线使用教程

# 通用语义分割产线介绍
语义分割是一种计算机视觉技术，旨在将图像中的每个像素分配到特定的类别，从而实现对图像内容的精细化理解。语义分割不仅要识别出图像中的物体类型，还要对每个像素进行分类，这样使得同一类别的区域能够被完整标记。例如，在一幅街景图像中，语义分割可以将行人、汽车、天空和道路等不同类别的部分逐像素区分开来，形成一个详细的标签图。这项技术广泛应用于自动驾驶、医学影像分析和人机交互等领域，通常依赖于深度学习模型（如FCN、U-Net等），通过卷积神经网络（CNN）来提取特征并实现高精度的像素级分类，从而为进一步的智能分析提供基础。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3bc26c2d3642415db8516b105a1cf010&docGuid=5UTuma9YF5rEBe "")
**通用****语义分割****产线中包含了****语义分割****模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>

|模型名称|mloU（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|Deeplabv3_Plus-R50 |80.36|61.0531|1513.58|94.9 M|
|Deeplabv3_Plus-R101|81.10|100.026|2460.71|162.5 M|
|Deeplabv3-R50|79.90|82.2631|1735.83|138.3 M|
|Deeplabv3-R101|80.85|121.492|2685.51|205.9 M|
|OCRNet_HRNet-W18|80.67|48.2335|906.385|43.1 M|
|OCRNet_HRNet-W48|82.15|78.9976|2226.95|249.8 M|
|PP-LiteSeg-T|73.10|7.6827|138.683|28.5 M|
|PP-LiteSeg-B|75.25|-|-|47.0 M|
|SegFormer-B0 (slice)|76.73|11.1946|268.929|13.2 M|
|SegFormer-B1 (slice)|78.35|17.9998|403.393|48.5 M|
|SegFormer-B2 (slice)|81.60|48.0371|1248.52|96.9 M|
|SegFormer-B3 (slice)|82.47|64.341|1666.35|167.3 M|
|SegFormer-B4 (slice)|82.38|82.4336|1995.42|226.7 M|
|SegFormer-B5 (slice)|82.58|97.3717|2420.19|229.7 M|
**注：以上精度指标为 **[Cityscapes](https://www.cityscapes-dataset.com/)** 数据集 mloU。**

|模型名称|mloU（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|SeaFormer_base(slice)|40.92|24.4073|397.574|30.8 M|
|SeaFormer_large (slice)|43.66|27.8123|550.464|49.8 M|
|SeaFormer_small (slice)|38.73|19.2295|358.343|14.3 M|
|SeaFormer_tiny (slice)|34.58|13.9496|330.132|6.1M |
**注：以上精度指标为 **[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)** 数据集, slice 表示对输入图像进行了切图操作。**

</details>

# 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用语义分割产线的效果，也可以在本地使用命令行或 Python 体验通用语义分割产线的效果。

## 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/100062/webUI?source=appCenter)通用语义分割产线的效果，用官方提供的 Demo 图片进行识别，例如：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=d6a55347c3ed48dbab0a6172ac35087d&docGuid=5UTuma9YF5rEBe "")
如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

## 2.2 本地体验
在本地使用通用语义分割产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.2.1 命令行方式体验
一行命令即可快速体验语义分割产线效果

```
paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为目标检测产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```
执行后，将提示选择语义分割产线配置文件保存路径，默认保存至*当前目录*，也可 *自定义路径*。

此外，也可在执行命令时加入 -y 参数，则可跳过路径选择，直接将产线配置文件保存至当前目录。

获取产线配置文件后，可将 --pipeline 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 ./semantic_segmentation.yaml，只需执行：

```
paddlex --pipeline ./semantic_segmentation.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png
```
其中，--model、--device 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

运行后，得到的结果为：

```
{'img_path': '/root/.paddlex/predict_input/general_object_detection_002.png'}
```


### 2.2.2 Python脚本方式集成 
几行代码即可完成产线的快速推理，以通用语义分割产线为例：

```
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="semantic_segmentation")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_img("./output/") # 保存结果可视化图像
    res.save_to_json("./output/") # 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

* 实例化 `create_pipeline` 实例化语义分割产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|pipeline|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|str|无|
|device|产线模型推理设备。支持：“gpu”，“cpu”。|str|gpu|
|enable_hpi|是否启用高性能推理，仅当该产线支持高性能推理时可用。|bool|False|
* 调用语义分割产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

|参数类型|参数说明|
|-|-|
|Python Var|支持直接传入Python变量，如numpy.ndarray表示的图像数据；|
|str|支持传入待预测数据文件路径，如图像文件的本地路径：/root/data/img.jpg；|
|str|支持传入待预测数据文件url，如图像文件的网络url：https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png；|
|str|支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：/root/data/；|
|dict|支持传入字典类型，字典的key需要与具体产线对应，如图像分类产线为"img"，字典的val支持上述类型数据，如：{"img": "/root/data1"}；|
|list|支持传入列表，列表元素需为上述类型数据，如[numpy.ndarray, numpy.ndarray, ]，["/root/data/img1.jpg", "/root/data/img2.jpg", ]，["/root/data1", "/root/data2", ]，[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}, ]；|
* 调用 predict 方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以 batch 为单位对数据进行预测，因此预测结果为 list 形式表示的一组预测结果
* 对预测结果进行处理：每个样本的预测结果均为 dict 类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

|方法|说明|方法参数|
|-|-|-|
|print|打印结果到终端|format_json：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；|
|||indent：int类型，json格式化设置，仅当format_json为True时有效，默认为4；|
|||ensure_ascii：bool类型，json格式化设置，仅当format_json为True时有效，默认为False；|
|save_to_json|将结果保存为json格式的文件|save_path：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|
|||indent：int类型，json格式化设置，默认为4;|
|||ensure_ascii：bool类型，json格式化设置，默认为False；|
|save_to_img|将结果保存为图像格式的文件|save_path：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|

在执行上述 Python 脚本时，加载的是默认的语义分割产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

```
paddlex --get_pipeline_config semantic_segmentation
```
执行后，语义分割产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为* ./my_path*）：

```
paddlex --get_pipeline_config semantic_segmentation --config_save_path ./my_path
```
获取配置文件后，您即可对语义分割产线各项配置进行自定义，只需要修改 create_pipeline 方法中的 pipeline 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 *./my_path/*semantic_segmentation*.yaml* ，则只需执行：

```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/semantic_segmentation.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_img("./output/") # 保存结果可视化图像
    res.save_to_json("./output/") # 保存预测的结构化输出
```
# 开发集成/部署
如果通用语义分割产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用语义分割产线直接应用在您的 Python 项目中，可以参考 2.2.2 Python脚本方式中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

* 高性能部署：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX 高性能部署指南](../../../pipeline_deploy/high_performance_deploy.md)。
* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX 服务化部署指南](../../../pipeline_deploy/service_deploy.md)。
* 端侧部署：端侧部署是将模型部署在包括移动端、嵌入式以及边缘端在内的多种硬件平台。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

# 二次开发
如果通用语义分割产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用语义分割产线的在您的场景中的识别效果。

## 4.1 模型微调
由于通用语义分割产线包含语义分割模块，如果模型产线的效果不及预期，那么您需要参考[语义分割模块开发教程](../../../module_usage/tutorials/cv_modules/semantic_segmentation.md)中的**二次开发**章节（github可以直接链接标题），使用您的私有数据集对语义分割模型进行微调。

## 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  model: PP-LiteSeg-T  #可修改为微调后模型的本地路径
  device: "gpu"
  batch_size: 1
......
```
随后， 参考 *2.2 本地体验* 中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

#  多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 ****--device**** 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行语义分割产线的推理，使用的 Python 命令为：

```
paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 --device 修改为 npu 即可：

```
paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device npu:0
```
若您想在更多种类的硬件上使用通用语义分割产线，请参考[PaddleX多硬件使用指南](../../../installation/installation_other_devices.md)。