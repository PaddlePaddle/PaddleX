通用目标检测产线使用教程

# 通用目标检测产线介绍
目标检测旨在识别图像或视频中多个对象的类别及其位置，通过生成边界框来标记这些对象。与简单的图像分类不同，目标检测不仅需要识别出图像中有哪些物体，例如人、车和动物等，还需要准确地确定每个物体在图像中的具体位置，通常以矩形框的形式表示。该技术广泛应用于自动驾驶、监控系统和智能相册等领域，依赖于深度学习模型（如YOLO、Faster R-CNN等），这些模型能够高效地提取特征并进行实时检测，显著提升了计算机对图像内容理解的能力。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f161bf278aed4eefb6a6c8b5a66ec8aa&docGuid=2iPv0uRuAO3FBK "")
**通用****目标检测****产线中包含了****目标检测****模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>

|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|Cascade-FasterRCNN-ResNet50-FPN|41.1|-|-|245.4 M|
|Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN|45.0|-|-|246.2 M|
|CenterNet-DLA-34|37.6|-|-|75.4 M|
|CenterNet-ResNet50|38.9|-|-|319.7 M|
|DETR-R50|42.3|59.2132|5334.52|159.3 M|
|FasterRCNN-ResNet34-FPN|37.8|-|-|137.5 M|
|FasterRCNN-ResNet50-FPN|38.4|-|-|148.1 M|
|FasterRCNN-ResNet50-vd-FPN|39.5|-|-|148.1 M|
|FasterRCNN-ResNet50-vd-SSLDv2-FPN|41.4|-|-|148.1 M|
|FasterRCNN-ResNet50|36.7|-|-|120.2 M|
|FasterRCNN-ResNet101-FPN|41.4|-|-|216.3 M|
|FasterRCNN-ResNet101|39.0|-|-|188.1 M|
|FasterRCNN-ResNeXt101-vd-FPN|43.4|-|-|360.6 M|
|FasterRCNN-Swin-Tiny-FPN|42.6|-|-|159.8 M|
|FCOS-ResNet50|39.6|103.367|3424.91|124.2 M|
|PicoDet-L|42.6|16.6715|169.904|20.9 M|
|PicoDet-M|37.5|16.2311|71.7257|16.8 M|
|PicoDet-S|29.1|14.097|37.6563|4.4 M |
|PicoDet-XS|26.2|13.8102|48.3139|5.7M |
|PP-YOLOE_plus-L|52.9|33.5644|814.825|185.3 M|
|PP-YOLOE_plus-M|49.8|19.843|449.261|83.2 M|
|PP-YOLOE_plus-S|43.7|16.8884|223.059|28.3 M|
|PP-YOLOE_plus-X|54.7|57.8995|1439.93|349.4 M|
|RT-DETR-H|56.3|114.814|3933.39|435.8 M|
|RT-DETR-L|53.0|34.5252|1454.27|113.7 M|
|RT-DETR-R18|46.5|19.89|784.824|70.7 M|
|RT-DETR-R50|53.1|41.9327|1625.95|149.1 M|
|RT-DETR-X|54.8|61.8042|2246.64|232.9 M|
|YOLOv3-DarkNet53|39.1|40.1055|883.041|219.7 M|
|YOLOv3-MobileNetV3|31.4|18.6692|267.214|83.8 M|
|YOLOv3-ResNet50_vd_DCN|40.6|31.6276|856.047|163.0 M|
|YOLOX-L|50.1|185.691|1250.58|192.5 M|
|YOLOX-M|46.9|123.324|688.071|90.0 M|
|YOLOX-N|26.1|79.1665|155.59|3.4M|
|YOLOX-S|40.4|184.828|474.446|32.0 M|
|YOLOX-T|32.9|102.748|212.52|18.1 M|
|YOLOX-X|51.8|227.361|2067.84|351.5 M|
**注**：**以上精度指标为**[COCO2017](https://cocodataset.org/#home)**验证集 mAP(0.5:0.95)。****所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

</details>

# 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用目标检测产线的效果，也可以在本地使用命令行或 Python 体验通用目标检测产线的效果。

## 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/70230/webUI)通用目标检测产线的效果，用官方提供的 demo 图片进行识别，例如：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=1eaf75c31eb34ff7a931da55a3d1fb02&docGuid=2iPv0uRuAO3FBK "")
如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

## 2.2 本地体验
在本地使用通用目标检测产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.2.1 命令行方式体验
一行命令即可快速体验目标检测产线效果

```
paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为目标检测产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```
执行后，将提示选择目标检测产线配置文件保存路径，默认保存至*当前目录*，也可 *自定义路径*。

此外，也可在执行命令时加入 -y 参数，则可跳过路径选择，直接将产线配置文件保存至当前目录。

获取产线配置文件后，可将 --pipeline 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 ./object_detection.yaml，只需执行：

```
paddlex --pipeline ./object_detection.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png
```
其中，--model、--device 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

运行后，得到的结果为：

```
{'img_path': '/root/.paddlex/predict_input/general_object_detection_002.png', 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188097476959229, 'coordinate': [661, 93, 870, 305]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7743489146232605, 'coordinate': [76, 274, 330, 520]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7270504236221313, 'coordinate': [285, 94, 469, 297]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5570532083511353, 'coordinate': [310, 361, 685, 712]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5484835505485535, 'coordinate': [764, 285, 924, 440]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5160726308822632, 'coordinate': [853, 169, 987, 303]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.5142655968666077, 'coordinate': [0, 0, 1072, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5101479291915894, 'coordinate': [57, 23, 213, 176]}]}
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=a14adac98f61435296164a77d5a32f7b&docGuid=2iPv0uRuAO3FBK "")
### 2.2.2 Python脚本方式集成 
几行代码即可完成产线的快速推理，以通用目标检测产线为例：

```
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="object_detection")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_img("./output/") # 保存结果可视化图像
    res.save_to_json("./output/") # 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

* 实例化 `create_pipeline` 实例化目标检测产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|pipeline|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|str|无|
|device|产线模型推理设备。支持：“gpu”，“cpu”。|str|gpu|
|enable_hpi|是否启用高性能推理，仅当该产线支持高性能推理时可用。|bool|False|
* 调用目标检测产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

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

在执行上述 Python 脚本时，加载的是默认的目标检测产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

```
paddlex --get_pipeline_config object_detection
```
执行后，目标检测产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为* ./my_path*）：

```
paddlex --get_pipeline_config object_detection --config_save_path ./my_path
```
获取配置文件后，您即可对目标检测产线各项配置进行自定义，只需要修改 create_pipeline 方法中的 pipeline 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 *./my_path/*object_detection*.yaml* ，则只需执行：

```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_img("./output/") # 保存结果可视化图像
    res.save_to_json("./output/") # 保存预测的结构化输出
```
# 开发集成/部署
如果通用目标检测产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用目标检测产线直接应用在您的 Python 项目中，可以参考 2.2.2 Python脚本方式中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

* 高性能部署：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX 高性能部署指南](../../../pipeline_deploy/high_performance_deploy.md)。
* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX 服务化部署指南](../../../pipeline_deploy/service_deploy.md)。
* 端侧部署：端侧部署是将模型部署在包括移动端、嵌入式以及边缘端在内的多种硬件平台。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

# 二次开发
如果通用目标检测产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用目标检测产线的在您的场景中的识别效果。

## 4.1 模型微调
由于通用目标检测产线包含目标检测模块，如果模型产线的效果不及预期，那么您需要参考[目标检测模块开发教程](../../../module_usage/tutorials/cv_modules/object_detection.md)中的**二次开发**章节，使用您的私有数据集对目标检测模型进行微调。

## 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  model: PicoDet-S  #可修改为微调后模型的本地路径
  device: "gpu"
  batch_size: 1
......
```
随后， 参考 *2.2 本地体验* 中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

#  多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 ****--device**** 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行目标检测产线的推理，使用的 Python 命令为：

```
paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 --device 修改为 npu 即可：

```
paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device npu:0
```
若您想在更多种类的硬件上使用通用目标检测产线，请参考[PaddleX多硬件使用指南](../../../installation/installation_other_devices.md)。