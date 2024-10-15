简体中文 | [English](seal_recognition_en.md)

# 印章文本识别产线使用教程

## 1. 印章文本识别产线介绍
印章文本识别是一种自动从文档或图像中提取和识别印章内容的技术，印章文本的识别是文档处理的一部分，在很多场景都有用途，例如合同比对，出入库审核以及发票报销审核等场景。


![](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/PP-ChatOCRv3_doc_seal/01.png)


**印章文本识别**产线中包含版面区域分析模块、印章印章文本检测模块和文本识别模块。

**如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>



**版面区域分析模块模型：**

|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M）|
|-|-|-|-|-|
|PicoDet-L_layout_3cls|89.3|15.7425|159.771|22.6 M|
|RT-DETR-H_layout_3cls|95.9|114.644|3832.62|470.1M|
|RT-DETR-H_layout_17cls|92.6|115.126|3827.25|470.2M|

**注：以上精度指标的评估集是 PaddleX 自建的版面区域分析数据集，包含 1w 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

**印章文本检测模块模型：**

|模型|检测Hmean（%）|GPU推理耗时（ms）|CPU推理耗时 (ms)|模型存储大小（M)|介绍|
|-|-|-|-|-|-|
|PP-OCRv4_server_seal_det|98.21|84.341|2425.06|109|PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署|
|PP-OCRv4_mobile_seal_det|96.47|10.5878|131.813|4.6|PP-OCRv4的移动端印章文本检测模型，效率更高，适合在端侧部署|

**注：以上精度指标的评估集是自建的数据集，包含500张圆形印章图像。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。**

**文本识别模块模型：**

|模型名称|识别Avg Accuracy(%)|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M）|
|-|-|-|-|-|
|PP-OCRv4_mobile_rec |78.20|7.95018|46.7868|10.6 M|
|PP-OCRv4_server_rec |79.20|7.19439|140.179|71.2 M|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集 ，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

</details>

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验印章文本识别产线的效果，也可以在本地使用命令行或 Python 体验印章文本识别产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/182491/webUI)文档场景信息抽取v3产线中的印章文本识别的效果，用官方提供的 Demo 图片进行识别，例如：

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/02.png)

如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

### 2.2 本地体验
在本地使用印章文本识别产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验印章文本识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png)，并将 `--input` 替换为本地路径，进行预测

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device gpu:0 --save_path ./output
```
参数说明：

```
--pipeline：产线名称，此处为印章文本识别产线
--input：待处理的输入图片的本地路径或URL
--device: 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
--save_path: 输出结果保存路径
```

在执行上述 Python 脚本时，加载的是默认的印章文本识别产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details>
   <summary> 👉点击展开</summary>

```
paddlex --get_pipeline_config seal_recognition
```
执行后，印章文本识别产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 `./my_path` ）：

```
paddlex --get_pipeline_config seal_recognition --save_path ./my_path
```

获取产线配置文件后，可将 `--pipeline` 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 `./seal_recognition.yaml`，只需执行：

```
paddlex --pipeline seal_recognition.yaml --input seal_text_det.png --save_path ./output
```
其中，`--model`、`--device` 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

</details>

运行后，得到的结果为：

<details>
   <summary> 👉点击展开</summary>

```
{'input_path': 'seal_text_det.png', 'layout_result': {'input_path': 'seal_text_det.png', 'boxes': [{'cls_id': 2, 'label': 'seal', 'score': 0.9813116192817688, 'coordinate': [0, 5.2238655, 639.59766, 637.6985]}]}, 'ocr_result': [{'input_path': PosixPath('/root/.paddlex/temp/tmp19fn93y5.png'), 'dt_polys': [array([[468, 469],
       [472, 469],
       [477, 471],
       [507, 501],
       [509, 505],
       [509, 509],
       [508, 513],
       [506, 514],
       [456, 553],
       [454, 555],
       [391, 581],
       [388, 581],
       [309, 590],
       [306, 590],
       [234, 577],
       [232, 577],
       [172, 548],
       [170, 546],
       [121, 504],
       [118, 501],
       [118, 496],
       [119, 492],
       [121, 490],
       [152, 463],
       [156, 461],
       [160, 461],
       [164, 463],
       [202, 495],
       [252, 518],
       [311, 530],
       [371, 522],
       [425, 501],
       [464, 471]]), array([[442, 439],
       [445, 442],
       [447, 447],
       [449, 490],
       [448, 494],
       [446, 497],
       [440, 499],
       [197, 500],
       [193, 499],
       [190, 496],
       [188, 491],
       [188, 448],
       [189, 444],
       [192, 441],
       [197, 439],
       [438, 438]]), array([[465, 341],
       [470, 344],
       [472, 346],
       [476, 356],
       [476, 419],
       [475, 424],
       [472, 428],
       [467, 431],
       [462, 433],
       [175, 434],
       [170, 433],
       [166, 430],
       [163, 426],
       [161, 420],
       [161, 354],
       [162, 349],
       [165, 345],
       [170, 342],
       [175, 340],
       [460, 340]]), array([[326,  34],
       [481,  85],
       [485,  88],
       [488,  90],
       [584, 220],
       [586, 225],
       [587, 229],
       [589, 378],
       [588, 383],
       [585, 388],
       [581, 391],
       [576, 393],
       [570, 392],
       [507, 373],
       [502, 371],
       [498, 367],
       [496, 359],
       [494, 255],
       [423, 162],
       [322, 129],
       [246, 151],
       [205, 169],
       [144, 252],
       [139, 360],
       [137, 365],
       [134, 369],
       [128, 373],
       [ 66, 391],
       [ 61, 392],
       [ 56, 390],
       [ 51, 387],
       [ 48, 382],
       [ 47, 377],
       [ 49, 230],
       [ 50, 225],
       [ 52, 221],
       [149,  89],
       [153,  86],
       [157,  84],
       [318,  34],
       [322,  33]])], 'dt_scores': [0.9943362380813267, 0.9994290391836306, 0.9945320407374245, 0.9908104427126033], 'rec_text': ['5263647368706', '吗繁物', '发票专用章', '天津君和缘商贸有限公司'], 'rec_score': [0.9921098351478577, 0.997374951839447, 0.9999369382858276, 0.9901710152626038]}]}
```
</details>

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/03.png)

可视化图片默认保存在 `output` 目录下，您也可以通过 `--save_path` 进行自定义。


### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以印章文本识别产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="seal_recognition")

output = pipeline.predict("seal_text_det.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存可视化结果
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`pipeline`|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|`str`|无|
|`device`|产线模型推理设备。支持：“gpu”，“cpu”。|`str`|`gpu`|
|`enable_hpi`|是否启用高性能推理，仅当该产线支持高性能推理时可用。|`bool`|`False`|

（2）调用产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型      | 参数说明                                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | 支持直接传入Python变量，如numpy.ndarray表示的图像数据。                                               |
| str         | 支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`。                                   |
| str           | 支持传入待预测数据文件URL，如图像文件的网络URL：[示例](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png)。|
| str           | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`。                               |
| dict          | 支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：`{\"img\": \"/root/data1\"}`。|
| list          | 支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]`，`[\"/root/data1\", \"/root/data2\"]`，`[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]`。|

（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：


| 方法         | 说明                        | 方法参数                                                                                               |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | 打印结果到终端              | `- format_json`：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br>`- indent`：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，仅当format_json为True时有效，默认为False； |
| save_to_json | 将结果保存为json格式的文件   | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br>`- indent`：int类型，json格式化设置，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，默认为False； |
| save_to_img  | 将结果保存为图像格式的文件  | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致； |

若您获取了配置文件，即可对印章文本识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/seal_recognition.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/seal_recognition.yaml")
output = pipeline.predict("seal_text_det.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存可视化结果
```
## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 **高性能部署**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX高性能部署指南](../../../pipeline_deploy/high_performance_deploy.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

下面是API参考和多语言服务调用示例：



📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果印章文本识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升印章文本识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于印章文本识别产线包含三个模块，模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，参考如下规则进行分析和模型微调：

* 印章区域在整体版面中定位错误，那么可能是版面区域定位模块存在不足，您需要参考[版面区域检测模块开发教程](../../../module_usage/tutorials/ocr_modules/layout_detection.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/layout_detection.md#四二次开发)章节，使用您的私有数据集对版面区域定位模型进行微调。
* 有较多的文本未被检测出来（即文本漏检现象），那么可能是文本检测模型存在不足，您需要参考[印章文本检测模块开发教程](../../../module_usage/tutorials/ocr_modules/seal_text_detection.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/seal_text_detection.md#四二次开发)章节，使用您的私有数据集对文本检测模型进行微调。
* 已检测到的文本中出现较多的识别错误（即识别出的文本内容与实际文本内容不符），这表明文本识别模型需要进一步改进，您需要参考[文本识别模块开发教程](../../../module_usage/tutorials/ocr_modules/text_recognition.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/text_recognition.md#四二次开发)章节对文本识别模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```python
......
 Pipeline:
  layout_model: RT-DETR-H_layout_3cls #可修改为微调后模型的本地路径
  text_det_model: PP-OCRv4_server_seal_det  #可修改为微调后模型的本地路径
  text_rec_model: PP-OCRv4_server_rec #可修改为微调后模型的本地路径
  layout_batch_size: 1
  text_rec_batch_size: 1
  device: "gpu:0"
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持

PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 `--device` 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行印章文本识别产线的推理，使用的 Python 命令为：

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device gpu:0 --save_path output
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 `--device` 修改为npu 即可：

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device npu:0 --save_path output
```
若您想在更多种类的硬件上使用印章文本识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/installation_other_devices.md)。
