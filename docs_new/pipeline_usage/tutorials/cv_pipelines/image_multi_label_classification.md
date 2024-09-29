# 通用图像多标签分类产线使用教程

## 通用图像多标签分类产线介绍
图像多标签分类是一种将一张图像同时分配到多个相关类别的技术，广泛应用于图像标注、内容推荐和社交媒体分析等领域。它能够识别图像中存在的多个物体或特征，例如一张图片中同时包含“狗”和“户外”这两个标签。通过使用深度学习模型，图像多标签分类能够自动提取图像特征并进行准确分类，以便为用户提供更加全面的信息。这项技术在智能搜索引擎和自动内容生成等应用中具有重要意义。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b9bccdc55966461fa918d96fd911e9c3&docGuid=L6gv05IXH5vXSR "")
**通用图像多标签分类产线中包含了图像多标签分类模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。



<details>
   <summary> 👉模型列表详情</summary>

|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|CLIP_vit_base_patch16_448_ML|89.15|-|-|325.6|
|PP-HGNetV2-B0_ML|80.98|-|-|39.6|
|PP-HGNetV2-B4_ML|87.96|-|-|88.5|
|PP-HGNetV2-B6_ML|91.25|-|-|286.5|
|PP-LCNet_x1_0_ML|77.96|-|-|29.4|
|ResNet50_ML|83.50|-|-|108.9|
**注：以上精度指标为 **[COCO2017](https://cocodataset.org/#home)** 的多标签分类任务****mAP。**
</details>


## 快速开始
PaddleX 支持在本地使用命令行或 Python 体验通用图像多标签分类产线的效果。

在本地使用通用图像多标签分类产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验图像多标签分类产线效果 

```
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为图像多标签分类产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```
执行后，将提示选择图像多标签分类产线配置文件保存路径，默认保存至*当前目录*，也可 *自定义路径*。

此外，也可在执行命令时加入 -y 参数，则可跳过路径选择，直接将产线配置文件保存至当前目录。

获取产线配置文件后，可将 --pipeline 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 ./multi_label_image_classification.yaml，只需执行：

```
paddlex --pipeline ./multi_label_image_classification.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg
```
其中，--model、--device 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

运行后，得到的结果为：

```
{'img_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [21, 0, 30, 24], 'scores': [0.99257, 0.70596, 0.63001, 0.57852], 'label_names': ['bear', 'person', 'skis', 'backpack']}
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f6772e98c9df4c428d424531618be0e9&docGuid=L6gv05IXH5vXSR "")
### 2.2 Python脚本方式集成 
几行代码即可完成产线的快速推理，以通用图像多标签分类产线为例：

```
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multi_label_image_classification")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

* 实例化 `create_pipeline` 实例化图像多标签分类产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|pipeline|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|str|无|
|device|产线模型推理设备。支持：“gpu”，“cpu”。|str|gpu|
|enable_hpi|是否启用高性能推理，仅当该产线支持高性能推理时可用。|bool|False|
* 调用图像多标签分类产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

|参数类型|参数说明|
|-|-|
|Python Var|支持直接传入Python变量，如numpy.ndarray表示的图像数据；|
|str|支持传入待预测数据文件路径，如图像文件的本地路径：/root/data/img.jpg；|
|str|支持传入待预测数据文件url，如图像文件的网络url：https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg；|
|str|支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：/root/data/；|
|dict|支持传入字典类型，字典的key需要与具体产线对应，如图像多标签分类产线为"img"，字典的val支持上述类型数据，如：

{"img": "/root/data1"}；|
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

在执行上述 Python 脚本时，加载的是默认的图像多标签分类产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

```
paddlex --get_pipeline_config multi_label_image_classification
```
执行后，图像多标签分类产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为* ./my_path*）：

```
paddlex --get_pipeline_config multi_label_image_classification --config_save_path ./my_path
```
获取配置文件后，您即可对图像多标签分类产线各项配置进行自定义，只需要修改 create_pipeline 方法中的 pipeline 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 *./my_path/*multi_label_image_classification*.yaml* ，则只需执行：

```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/multi_label_image_classification.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
## 开发集成/部署
如果通用图像多标签分类产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用图像多标签分类产线直接应用在您的 Python 项目中，可以参考 2.2 Python脚本方式中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

* 高性能部署：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX 高性能部署指南](../../../pipeline_deploy/high_performance_deploy.md)。
* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX 服务化部署指南](../../../pipeline_deploy/service_deploy.md)。
* 端侧部署：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 二次开发
如果通用图像多标签分类产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用图像多标签分类产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用图像多标签分类产线包含图像多标签分类模块，如果模型产线的效果不及预期，那么您需要参考[图像多标签分类模块开发教程](../../../module_usage/tutorials/cv_modules/ml_classification.md)中的**二次开发**章节，使用您的私有数据集对图像多标签分类模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
 Pipeline:
  model: PP-LCNet_x1_0_ML   #可修改为微调后模型的本地路径
  batch_size: 1
  device: "gpu:0"
......
```
随后， 参考  *2.2 本地体验 * 中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 ****--device**** 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行图像多标签分类产线的推理，使用的 Python 命令为：

```
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 --device 修改为 npu 即可：

```
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device npu:0
```
若您想在更多种类的硬件上使用通用图像多标签分类产线，请参考[PaddleX多硬件使用指南](../../../installation/installation_other_devices.md)。