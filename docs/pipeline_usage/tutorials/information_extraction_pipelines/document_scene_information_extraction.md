简体中文 | [English](document_scene_information_extraction_en.md)

# 文档场景信息抽取v3产线使用教程

## 1. 文档场景信息抽取v3产线介绍
文档场景信息抽取v3（PP-ChatOCRv3）是飞桨特色的文档和图像智能分析解决方案，结合了 LLM 和 OCR 技术，一站式解决版面分析、生僻字、多页 pdf、表格、印章识别等常见的复杂文档信息抽取难点问题，结合文心大模型将海量数据和知识相融合，准确率高且应用广泛。

![](https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02)

文档场景信息抽取v3产线中包含**表格结构识别模块**、**版面区域检测模块**、**文本检测模块**、**文本识别模块**、**印章文本检测模块**、**文本图像矫正模块**、**文档图像方向分类模块**。

**如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。其中部分模型的 benchmark 如下：

<details>
   <summary> 👉模型列表详情</summary>

**表格结构识别模块模型：**

<table>
  <tr>
    <th>模型</th>
    <th>精度（%）</th>
    <th>GPU推理耗时 (ms)</th>
    <th>CPU推理耗时（ms）</th>
    <th>模型存储大小 (M)</th>
    <th>介绍</th>
  </tr>
  <tr>
    <td>SLANet</td>
    <td>59.52</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td>SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用CPU 友好型轻量级骨干网络PP-LCNet、高低层特征融合模块CSP-PAN、结构与位置信息对齐的特征解码模块SLA Head，大幅提升了表格结构识别的精度和推理速度。</td>
  </tr>
   <tr>
    <td>SLANet_plus</td>
    <td>63.69</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td>SLANet_plus 是百度飞桨视觉团队自研的表格结构识别模型SLANet的增强版。相较于SLANet，SLANet_plus 对无线表、复杂表格的识别能力得到了大幅提升，并降低了模型对表格定位准确性的敏感度，即使表格定位出现偏移，也能够较准确地进行识别。</td>
  </tr>
</table>

**注：以上精度指标测量PaddleX 内部自建英文表格识别数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**


**版面区域检测模块模型：**

|模型|mAP(0.5)（%）|GPU推理耗时（ms）|CPU推理耗时 (ms)|模型存储大小（M）|介绍|
|-|-|-|-|-|-|
|PicoDet_layout_1x|86.8|13.0|91.3|7.4|基于PicoDet-1x在PubLayNet数据集训练的高效率版面区域定位模型，可定位包含文字、标题、表格、图片以及列表这5类区域|
|PicoDet-S_layout_3cls|87.1|13.5 |45.8 |4.8|基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章|
|PicoDet-S_layout_17cls|70.3|13.6|46.2|4.8|基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章|
|PicoDet-L_layout_3cls|89.3|15.7|159.8|22.6|基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章|
|PicoDet-L_layout_17cls|79.9|17.2 |160.2|22.6|基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章|
|RT-DETR-H_layout_3cls|95.9|114.6|3832.6|470.1|基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含3个类别：表格，图像和印章|
|RT-DETR-H_layout_17cls|92.6|115.1|3827.2|470.2|基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章|


**注：以上精度指标的评估集是 PaddleOCR 自建的版面区域分析数据集，包含中英文论文、杂志和研报等常见的 1w 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。**

**文本检测模块模型：**

|模型|检测Hmean（%）|GPU推理耗时（ms）|CPU推理耗时 (ms)|模型存储大小（M)|介绍|
|-|-|-|-|-|-|
|PP-OCRv4_server_det|82.69|83.3501|2434.01|109|PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署|
|PP-OCRv4_mobile_det|77.79|10.6923|120.177|4.7|PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

**文本识别模块模型：**

<table >
    <tr>
        <th>模型</th>
        <th>识别 Avg Accuracy(%)</th>
        <th>GPU推理耗时（ms）</th>
        <th>CPU推理耗时 (ms)</th>
        <th>模型存储大小（M）</th>
        <th>介绍</th>
    </tr>
    <tr>
        <td>PP-OCRv4_mobile_rec</td>
        <td>78.20</td>
        <td>7.95018</td>
        <td>46.7868</td>
        <td>10.6 M</td>
        <td rowspan="2">PP-OCRv4是百度飞桨视觉团队自研的文本识别模型PP-OCRv3的下一个版本，通过引入数据增强方案、GTC-NRTR指导分支等策略，在模型推理速度不变的情况下，进一步提升了文本识别精度。该模型提供了服务端（server）和移动端（mobile）两个不同版本，来满足不同场景下的工业需求。</td>
    </tr>
    <tr>
        <td>PP-OCRv4_server_rec </td>
        <td>79.20</td>
        <td>7.19439</td>
        <td>140.179</td>
        <td>71.2 M</td>
    </tr>
</table>

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**


<table >
    <tr>
        <th>模型</th>
        <th>识别 Avg Accuracy(%)</th>
        <th>GPU推理耗时（ms）</th>
        <th>CPU推理耗时（ms）</th>
        <th>模型存储大小（M）</th>
        <th>介绍</th>
    </tr>
    <tr>
        <td>ch_SVTRv2_rec</td>
        <td>68.81</td>
        <td>8.36801</td>
        <td>165.706</td>
        <td>73.9 M</td>
        <td rowspan="1">
        SVTRv2 是一种由复旦大学视觉与学习实验室（FVL）的OpenOCR团队研发的服务端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，A榜端到端识别精度相比PP-OCRv4提升6%。
    </td>
    </tr>
</table>


**注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)A榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

<table >
    <tr>
        <th>模型</th>
        <th>识别 Avg Accuracy(%)</th>
        <th>GPU推理耗时（ms）</th>
        <th>CPU推理耗时（ms）</th>
        <th>模型存储大小（M）</th>
        <th>介绍</th>
    </tr>
    <tr>
        <td>ch_RepSVTR_rec</td>
        <td>65.07</td>
        <td>10.5047</td>
        <td>51.5647</td>
        <td>22.1 M</td>
        <td rowspan="1">    RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。</td>
    </tr>
</table>

**注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)B榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

**印章文本检测模块模型：**

|模型|检测Hmean（%）|GPU推理耗时（ms）|CPU推理耗时 (ms)|模型存储大小（M)|介绍|
|-|-|-|-|-|-|
|PP-OCRv4_server_seal_det|98.21|84.341|2425.06|109|PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署|
|PP-OCRv4_mobile_seal_det|96.47|10.5878|131.813|4.6|PP-OCRv4的移动端印章文本检测模型，效率更高，适合在端侧部署|

**注：以上精度指标的评估集是自建的数据集，包含500张圆形印章图像。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。**

**文本图像矫正模块模型：**

|模型|MS-SSIM （%）|模型存储大小（M)|介绍|
|-|-|-|-|
|UVDoc|54.40|30.3 M|高精度文本图像矫正模型|

**模型的精度指标测量自 [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html)。**

**文档图像方向分类模块模型：**

|模型|Top-1 Acc（%）|GPU推理耗时（ms）|CPU推理耗时 (ms)|模型存储大小（M)|介绍|
|-|-|-|-|-|-|
|PP-LCNet_x1_0_doc_ori|99.06|3.84845|9.23735|7|基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度|

**注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。**

</details>

****

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验文档场景信息抽取v3产线的效果，也可以在本地使用  Python 体验文档场景信息抽取v3产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/182491/webUI)文档场景信息抽取v3产线的效果，用官方提供的 Demo 图片进行识别，例如：

![](https://github.com/user-attachments/assets/aa261b2b-b79c-4487-9323-dfcc43c3d581)

如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

### 2.2 本地体验
在本地使用文档场景信息抽取v3产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

几行代码即可完成产线的快速推理，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf)，以通用文档场景信息抽取v3产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="PP-ChatOCRv3-doc",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""} # 请填入您的ak与sk，否则无法调用大模型
    # llm_params={"api_type": "aistudio", "access_token": ""} # 请填入您的access_token，否则无法调用大模型
    )

visual_result, visual_info = pipeline.visual_predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf")

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

vector = pipeline.build_vector(visual_info=visual_info)

chat_result = pipeline.chat(
    key_list=["乙方", "手机号"],
    visual_info=visual_info,
    vector=vector,
    )
chat_result.print()
```
**注**：目前仅支持文心大模型，支持在[百度云千帆平台](https://console.bce.baidu.com/qianfan/ais/console/onlineService)或者[星河社区 AIStudio](https://aistudio.baidu.com/)上获取相关的 ak/sk(access_token)。如果使用百度云千帆平台，可以参考[AK和SK鉴权调用API流程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Hlwerugt8) 获取ak/sk，如果使用星河社区 AIStudio，可以在[星河社区 AIStudio 访问令牌](https://aistudio.baidu.com/account/accessToken)中获取 access_token。

运行后，输出结果如下：

```
{'chat_res': {'乙方': '股份测试有限公司', '手机号': '19331729920'}, 'prompt': ''}
```

在上述 Python 脚本中，执行了如下四个步骤：

（1）调用 `create_pipeline` 方法实例化文档场景信息抽取v3产线对象，相关参数说明如下：

|参数|参数类型|默认值|参数说明|
|-|-|-|-|
|`pipeline`|str|无|产线名称或是产线配置文件路径，如为产线名称，则必须为 PaddleX 所支持的产线；|
|`llm_name`|str|"ernie-3.5"|大语言模型名称，目前支持`ernie-4.0`，`ernie-3.5`，更多模型支持中;|
|`llm_params`|dict|`{}`|LLM相关API配置；|
|`device`|str、None|`None`|运行设备（`None`为自动适配）；|

（2）调用文档场景信息抽取v3产线对象的 `visual_predict` 方法进行视觉推理预测，相关参数说明如下：

|参数|参数类型|默认值|参数说明|
|-|-|-|-|
|`input`|Python Var|无|用于输入待预测数据，支持直接传入Python变量，如`numpy.ndarray`表示的图像数据；|
|`input`|str|无|用于输入待预测数据，支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`；|
|`input`|str|无|用于输入待预测数据，支持传入待预测数据文件url，如`https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf`；|
|`input`|str|无|用于输入待预测数据，支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`；|
|`input`|dict|无|用于输入待预测数据，支持传入字典类型，字典的key需要与具体产线对应，如文档场景信息抽取v3产线为"img"，字典的val支持上述类型数据，如：`{"img": "/root/data1"}`；|
|`input`|list|无|用于输入待预测数据，支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]`，`["/root/data/img1.jpg", "/root/data/img2.jpg"]`，`["/root/data1", "/root/data2"]`，`[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`；|
|`use_doc_image_ori_cls_model`|bool|`True`|是否使用方向分类模型；|
|`use_doc_image_unwarp_model`|bool|`True`|是否使用版面矫正产线；|
|`use_seal_text_det_model`|bool|`True`|是否使用弯曲文本检测产线；|

（3）调用视觉推理预测结果对象的相关方法对视觉推理预测结果进行保存，具体方法如下：

|方法|参数|方法说明|
|-|-|-|
|`save_to_img`|`save_path`|将OCR预测结果、版面分析结果、表格识别结果保存为图片文件，参数`save_path`用于指定保存的路径；|
|`save_to_html`|`save_path`|将表格识别结果保存为html文件，参数`save_path`用于指定保存的路径；|
|`save_to_xlsx`|`save_path`|将表格识别结果保存为xlsx文件，参数`save_path`用于指定保存的路径；|

（4）调用文档场景信息抽取v3产线对象的 `chat` 方法与大模型进行交互，相关参数说明如下：

|参数|参数类型|默认值|参数说明|
|-|-|-|-|
|`key_list`|str|无|用于查询的关键字（query）；支持“，”或“,”作为分隔符的多个关键字组成的字符串，如“乙方，手机号”；|
|`key_list`|list|无|用于查询的关键字（query），支持`list`形式表示的一组关键字，其元素为`str`类型；|

在执行上述 Python 脚本时，加载的是默认的文档场景信息抽取v3产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

```
paddlex --get_pipeline_config PP-ChatOCRv3-doc
```

执行后，文档场景信息抽取v3产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 `./my_path` ）：

```
paddlex --get_pipeline_config PP-ChatOCRv3-doc --save_path ./my_path
```

获取配置文件后，您即可对文档场景信息抽取v3产线各项配置进行自定义：

```yaml
Pipeline:
  layout_model: RT-DETR-H_layout_3cls
  table_model: SLANet_plus
  text_det_model: PP-OCRv4_server_det
  text_rec_model: PP-OCRv4_server_rec
  seal_text_det_model: PP-OCRv4_server_seal_det
  doc_image_ori_cls_model: null
  doc_image_unwarp_model: null
  llm_name: "ernie-3.5"
  llm_params:
    api_type: qianfan
    ak:
    sk:
```

在上述配置中，您可以修改产线各模块加载的模型，也可以修改使用的大模型。各模块支持模型列表请参考模块文档，大模型支持列表为：ernie-4.0、ernie-3.5、ernie-3.5-8k、ernie-lite、ernie-tiny-8k、ernie-speed、ernie-speed-128k、ernie-char-8k。

修改后，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可应用配置。

例如，若您的配置文件保存在 `./my_path/PP-ChatOCRv3-doc.yaml` ，则只需执行：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="./my_path/PP-ChatOCRv3-doc.yaml",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""} # 请填入您的ak与sk，否则无法调用大模型
    # llm_params={"api_type": "aistudio", "access_token": ""} # 请填入您的access_token，否则无法调用大模型
    )

visual_result, visual_info = pipeline.visual_predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf")

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

vector = pipeline.build_vector(visual_info=visual_info)

chat_result = pipeline.chat(
    key_list=["乙方", "手机号"],
    visual_info=visual_info,
    vector=vector,
    )
chat_result.print()
```

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 本地体验](#22-本地体验)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 **高性能推理**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

下面是API参考和多语言服务调用示例：

<details>
<summary>API参考</summary>

对于服务提供的所有操作：

- 响应体以及POST请求的请求体均为JSON数据（JSON对象）。
- 当请求处理成功时，响应状态码为`200`，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。固定为`0`。|
    |`errorMsg`|`string`|错误说明。固定为`"Success"`。|

    响应体还可能有`result`属性，类型为`object`，其中存储操作结果信息。

- 当请求处理未成功时，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。与响应状态码相同。|
    |`errorMsg`|`string`|错误说明。|

服务提供的操作如下：

- **`analyzeImage`**

    使用计算机视觉模型对图像进行分析，获得OCR、表格识别结果等，并提取图像中的关键信息。

    `POST /chatocr-vision`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`file`|`string`|服务可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。对于超过10页的PDF文件，只有前10页的内容会被使用。|是|
        |`fileType`|`integer`|文件类型。`0`表示PDF文件，`1`表示图像文件。若请求体无此属性，则服务将尝试根据URL自动推断文件类型。|否|
        |`useImgOrientationCls`|`boolean`|是否启用文档图像方向分类功能。默认启用该功能。|否|
        |`useImgUnwrapping`|`boolean`|是否启用文本图像矫正功能。默认启用该功能。|否|
        |`useSealTextDet`|`boolean`|是否启用印章文本检测功能。默认启用该功能。|否|
        |`inferenceParams`|`object`|推理参数。|否|

        `inferenceParams`的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`maxLongSide`|`integer`|推理时，若文本检测模型的输入图像较长边的长度大于`maxLongSide`，则将对图像进行缩放，使其较长边的长度等于`maxLongSide`。|否|

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`visionResults`|`array`|使用计算机视觉模型得到的分析结果。数组长度为1（对于图像输入）或文档页数与10中的较小者（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中每一页的处理结果。|
        |`visionInfo`|`object`|图像中的关键信息，可用作其他操作的输入。|

        `visionResults`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`texts`|`array`|文本位置、内容和得分。|
        |`tables`|`array`|表格位置和内容。|
        |`inputImage`|`string`|输入图像。图像为JPEG格式，使用Base64编码。|
        |`ocrImage`|`string`|OCR结果图。图像为JPEG格式，使用Base64编码。|
        |`layoutImage`|`string`|版面区域检测结果图。图像为JPEG格式，使用Base64编码。|

        `texts`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`poly`|`array`|文本位置。数组中元素依次为包围文本的多边形的顶点坐标。|
        |`text`|`string`|文本内容。|
        |`score`|`number`|文本识别得分。|

        `tables`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`bbox`|`array`|表格位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。|
        |`html`|`string`|HTML格式的表格识别结果。|

- **`buildVectorStore`**

    构建向量数据库。

    `POST /chatocr-vector`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`visionInfo`|`object`|图像中的关键信息。由`analyzeImage`操作提供。|是|
        |`minChars`|`integer`|启用向量数据库的最小数据长度。|否|
        |`llmRequestInterval`|`number`|调用大语言模型API的间隔时间。|否|
        |`llmName`|`string`|大语言模型名称。|否|
        |`llmParams`|`object`|大语言模型API参数。|否|

        当前，`llmParams` 可以采用如下形式：

        ```json
        {
          "apiType": "qianfan",
          "apiKey": "{千帆平台API key}",
          "secretKey": "{千帆平台secret key}"
        }
        ```

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`vectorStore`|`string`|向量数据库序列化结果，可用作其他操作的输入。|

- **`retrieveKnowledge`**

    进行知识检索。

    `POST /chatocr-retrieval`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`keys`|`array`|关键词列表。|是|
        |`vectorStore`|`string`|向量数据库序列化结果。由`buildVectorStore`操作提供。|是|
        |`llmName`|`string`|大语言模型名称。|否|
        |`llmParams`|`object`|大语言模型API参数。|否|

        当前，`llmParams` 可以采用如下形式：

        ```json
        {
          "apiType": "qianfan",
          "apiKey": "{千帆平台API key}",
          "secretKey": "{千帆平台secret key}"
        }
        ```

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`retrievalResult`|`string`|知识检索结果，可用作其他操作的输入。|

- **`chat`**

    与大语言模型交互，利用大语言模型提炼关键信息。

    `POST /chatocr-vision`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`keys`|`array`|关键词列表。|是|
        |`visionInfo`|`object`|图像中的关键信息。由`analyzeImage`操作提供。|是|
        |`taskDescription`|`string`|提示词任务。|否|
        |`rules`|`string`|提示词规则。用于自定义信息抽取规则，例如规范输出格式。|否|
        |`fewShot`|`string`|提示词示例。|否|
        |`vectorStore`|`string`|向量数据库序列化结果。由`buildVectorStore`操作提供。|否|
        |`retrievalResult`|`string`|知识检索结果。由`retrieveKnowledge`操作提供。|否|
        |`returnPrompts`|`boolean`|是否返回使用的提示词。默认启用。|否|
        |`llmName`|`string`|大语言模型名称。|否|
        |`llmParams`|`object`|大语言模型API参数。|否|

        当前，`llmParams` 可以采用如下形式：

        ```json
        {
          "apiType": "qianfan",
          "apiKey": "{千帆平台API key}",
          "secretKey": "{千帆平台secret key}"
        }
        ```

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`chatResult`|`object`|关键信息抽取结果。|
        |`prompts`|`object`|使用的提示词。|

        `prompts`的属性如下：

        |名称|类型|含义|
        |-|-|-|
        |`ocr`|`string`|OCR提示词。|
        |`table`|`string`|表格提示词。|
        |`html`|`string`|HTML提示词。|

</details>

<details>
<summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>

```python
import base64
import pprint
import sys

import requests


API_BASE_URL = "http://0.0.0.0:8080"
API_KEY = "{千帆平台API key}"
SECRET_KEY = "{千帆平台secret key}"
LLM_NAME = "ernie-3.5"
LLM_PARAMS = {
    "apiType": "qianfan",
    "apiKey": API_KEY,
    "secretKey": SECRET_KEY,
}

file_path = "./demo.jpg"
keys = ["电话"]

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {
    "file": file_data,
    "fileType": 1,
    "useImgOrientationCls": True,
    "useImgUnwrapping": True,
    "useSealTextDet": True,
}
resp_vision = requests.post(url=f"{API_BASE_URL}/chatocr-vision", json=payload)
if resp_vision.status_code != 200:
    print(
        f"Request to chatocr-vision failed with status code {resp_vision.status_code}."
    )
    pprint.pp(resp_vision.json())
    sys.exit(1)
result_vision = resp_vision.json()["result"]

for i, res in enumerate(result_vision["visionResults"]):
    print("Texts:")
    pprint.pp(res["texts"])
    print("Tables:")
    pprint.pp(res["tables"])
    ocr_img_path = f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    layout_img_path = f"layout_{i}.jpg"
    with open(layout_img_path, "wb") as f:
        f.write(base64.b64decode(res["layoutImage"]))
    print(f"Output images saved at {ocr_img_path} and {layout_img_path}")

payload = {
    "visionInfo": result_vision["visionInfo"],
    "minChars": 200,
    "llmRequestInterval": 1000,
    "llmName": LLM_NAME,
    "llmParams": LLM_PARAMS,
}
resp_vector = requests.post(url=f"{API_BASE_URL}/chatocr-vector", json=payload)
if resp_vector.status_code != 200:
    print(
        f"Request to chatocr-vector failed with status code {resp_vector.status_code}."
    )
    pprint.pp(resp_vector.json())
    sys.exit(1)
result_vector = resp_vector.json()["result"]

payload = {
    "keys": keys,
    "vectorStore": result_vector["vectorStore"],
    "llmName": LLM_NAME,
    "llmParams": LLM_PARAMS,
}
resp_retrieval = requests.post(url=f"{API_BASE_URL}/chatocr-retrieval", json=payload)
if resp_retrieval.status_code != 200:
    print(
        f"Request to chatocr-retrieval failed with status code {resp_retrieval.status_code}."
    )
    pprint.pp(resp_retrieval.json())
    sys.exit(1)
result_retrieval = resp_retrieval.json()["result"]

payload = {
    "keys": keys,
    "visionInfo": result_vision["visionInfo"],
    "taskDescription": "",
    "rules": "",
    "fewShot": "",
    "vectorStore": result_vector["vectorStore"],
    "retrievalResult": result_retrieval["retrievalResult"],
    "returnPrompts": True,
    "llmName": LLM_NAME,
    "llmParams": LLM_PARAMS,
}
resp_chat = requests.post(url=f"{API_BASE_URL}/chatocr-chat", json=payload)
if resp_chat.status_code != 200:
    print(
        f"Request to chatocr-chat failed with status code {resp_chat.status_code}."
    )
    pprint.pp(resp_chat.json())
    sys.exit(1)
result_chat = resp_chat.json()["result"]
print("\nPrompts:")
pprint.pp(result_chat["prompts"])
print("Final result:")
print(result_chat["chatResult"])
```

**注**：请在 `API_KEY`、`SECRET_KEY` 处填入您的 API key 和 secret key。

</details>
</details>
<br/>

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用文档场景信息抽取v3产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用表格识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用文档场景信息抽取v3产线包含六个模块，模型产线的效果不及预期可能来自于其中任何一个模块（文本图像矫正模块暂不支持二次开发）。

您可以对识别效果差的图片进行分析，参考如下规则进行分析和模型微调：

* 检测到的表格结构错误（如行列识别错误、单元格位置错误），那么可能是表格结构识别模块存在不足，您需要参考[表格结构识别模块开发教程](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md)中的**二次开发**章节，使用您的私有数据集对表格结构识别模型进行微调。
* 版面中存在定位错误（例如对表格、印章的位置识别错误），那么可能是版面区域定位模块存在不足，您需要参考[版面区域检测模块开发教程](../../../module_usage/tutorials/ocr_modules/layout_detection.md)中的**二次开发**章节，使用您的私有数据集对版面区域定位模型进行微调。
* 有较多的文本未被检测出来（即文本漏检现象），那么可能是文本检测模型存在不足，您需要参考[文本检测模块开发教程](../../../module_usage/tutorials/ocr_modules/text_detection.md)中的**二次开发**章节，使用您的私有数据集对文本检测模型进行微调。
* 已检测到的文本中出现较多的识别错误（即识别出的文本内容与实际文本内容不符），这表明文本识别模型需要进一步改进，您需要参考[文本识别模块开发教程](../../../module_usage/tutorials/ocr_modules/text_recognition.md)中的**二次开发**章节对文本识别模型进行微调。
* 已检测到的印章文本出现较多的识别错误，这表明印章文本检测模块模型需要进一步改进，您需要参考[印章文本检测模块开发教程](../../../module_usage/tutorials/ocr_modules/)中的**二次开发**章节对印章文本检测模型进行微调。
* 含文字区域的文档或证件的方向存在较多的识别错误，这表明文档图像方向分类模型需要进一步改进，您需要参考[文档图像方向分类模块开发教程](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)中的**二次开发**章节对文档图像方向分类模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  layout_model: RT-DETR-H_layout_3cls  #可修改为微调后模型的本地路径
  table_model: SLANet_plus  #可修改为微调后模型的本地路径
  text_det_model: PP-OCRv4_server_det  #可修改为微调后模型的本地路径
  text_rec_model: PP-OCRv4_server_rec  #可修改为微调后模型的本地路径
  seal_text_det_model: PP-OCRv4_server_seal_det  #可修改为微调后模型的本地路径
  doc_image_ori_cls_model: null   #可修改为微调后模型的本地路径
  doc_image_unwarp_model: null   #可修改为微调后模型的本地路径
......
```

随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU 和寒武纪 MLU 等多种主流硬件设备，**仅需设置 `device` 参数**即可完成不同硬件之间的无缝切换。

例如，使用文档场景信息抽取v3产线时，将运行设备从英伟达 GPU 更改为昇腾 NPU，仅需将脚本中的 `device` 修改为 npu 即可：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(
    pipeline="PP-ChatOCRv3-doc",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""},
    device="npu:0" # gpu:0 --> npu:0
    )
```
若您想在更多种类的硬件上使用通用文档场景信息抽取产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
