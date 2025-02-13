---
comments: true
---

# 行人属性识别产线使用教程

## 1. 行人属性识别产线介绍
行人属性识别是计算机视觉系统中的关键功能，用于在图像或视频中定位并标记行人的特定特征，如性别、年龄、衣物颜色和款式等。该任务不仅要求准确检测出行人，还需识别每个行人的详细属性信息。行人属性识别产线是定位并识别行人属性的端到端串联系统，广泛应用于智慧城市和安防监控等领域，可显著提升系统的智能化水平和管理效率。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/pedestrian_attribute.png"/>
<b>行人属性识别产线中包含了行人检测模块和行人属性识别模块</b>，每个模块中包含了若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。


<p><b>行人检测模块：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-YOLOE-L_human</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_human_pretrained.pdparams">训练模型</a></td>
<td>48.0</td>
<td>81.9</td>
<td>33.27 / 9.19</td>
<td>173.72 / 173.72</td>
<td>196.02</td>
<td rowspan="2">基于PP-YOLOE的行人检测模型</td>
</tr>
<tr>
<td>PP-YOLOE-S_human</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_human_pretrained.pdparams">训练模型</a></td>
<td>42.5</td>
<td>77.9</td>
<td>9.94 / 3.42</td>
<td>54.48 / 46.52</td>
<td>28.79</td>
</tr>
</table>
<p><b>注：以上精度指标为CrowdHuman数据集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>行人属性识别模块：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_pedestrian_attribute</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams">训练模型</a></td>
<td>92.2</td>
<td>2.35 / 0.49</td>
<td>3.17 / 1.25</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_pedestrian_attribute 是一种基于PP-LCNet的轻量级行人属性识别模型，包含26个类别</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标为 PaddleX 内部自建数据集 mA。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>

## 2. 快速开始
PaddleX 所提供的模型产线均可以快速体验效果，你可以在星河社区线体验行人属性识别产线的效果，也可以在本地使用命令行或 Python 体验行人属性识别产线的效果。

### 2.1 在线体验

您可以[在线体验](https://aistudio.baidu.com/community/app/387978/webUI?source=appCenter)行人属性识别产线的效果，用官方提供的 Demo 图片进行识别，例如：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/pedestrian_attribute_recognition/ped_attr_aistudio.png"/>

如果您对产线运行的效果满意，可以直接进行集成部署。您可以选择从云端下载部署包，也可以参考[2.2节本地体验](#22-本地体验)中的方法进行本地部署。如果对效果不满意，您可以利用私有数据<b>对产线中的模型进行微调训练</b>。如果您具备本地训练的硬件资源，可以直接在本地开展训练；如果没有，星河零代码平台提供了一键式训练服务，无需编写代码，只需上传数据后，即可一键启动训练任务。

### 2.2 本地体验
在本地使用行人属性识别产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
一行命令即可快速体验行人属性识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline pedestrian_attribute_recognition --input pedestrian_attribute_002.jpg --device gpu:0 --save_path ./output/
```
相关的参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：

```bash
{'res': {'input_path': 'pedestrian_attribute_002.jpg', 'boxes': [{'labels': ['Trousers(长裤)', 'Age18-60(年龄在18-60岁之间)', 'LongCoat(长外套)', 'Side(侧面)'], 'cls_scores': array([0.99965, 0.99963, 0.98866, 0.9624 ]), 'det_score': 0.9795178771018982, 'coordinate': [87.24581, 322.5872, 546.2697, 1039.9852]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Front(面朝前)', 'Age18-60(年龄在18-60岁之间)'], 'cls_scores': array([0.99996, 0.99872, 0.93379, 0.71614]), 'det_score': 0.967143177986145, 'coordinate': [737.91626, 306.287, 1150.5961, 1034.2979]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Age18-60(年龄在18-60岁之间)', 'Side(侧面)'], 'cls_scores': array([0.99996, 0.99514, 0.98726, 0.96224]), 'det_score': 0.9645745754241943, 'coordinate': [399.45944, 281.9107, 869.5312, 1038.9962]}]}}
```

运行结果参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/01.jpg"/>

#### 2.2.2 Python脚本方式集成
* 上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="pedestrian_attribute_recognition")

output = pipeline.predict("pedestrian_attribute_002.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化行人属性识别产线对象，具体参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>config</code></td>
<td>产线具体的配置信息（如果和<code>pipeline</code>同时设置，优先级高于<code>pipeline</code>，且要求产线名和<code>pipeline</code>一致）。</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）调用行人属性识别产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code></li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_threshold</code></td>
<td>行人检测可视化阈值</td>
<td><code>float | None</code></td>
<td>
<ul>
<li><b>float</b>：如<code>0.5</code>， 表示过滤掉所有阈值小于<code>0.5</code>的目标框；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>0.5</code>；</li>
</ul>
</td>
<td><code>0.5</code></td>
</tr>
<tr>
<td><code>cls_threshold</code></td>
<td>行人属性预测阈值</td>
<td><code>float | dict | list｜ None</code></td>
<td>
<ul>
<li><b>float</b>：表示属性识别的统一阈值；</li>
<li><b>list</b>：如<code>[0.5, 0.45, 0.48, 0.4]</code>，表示按照<code>label list</code>顺序的不同类别阈值；；</li>
<li><b>dict</b>：字典的key 为 <code>default</code> 和 <code>int</code> 类型，val 为 <code>float</code> 类型阈值，如<code>{"default": 0.5, 0: 0.45, 2: 0.48, 7: 0.4}</code>，<code>default</code> 表示多标签分类的统一阈值，其他 <code>int</code> 类型表示对 cls_id 为0的类别应用阈值 0.45、cls_id 为 1 的类别应用阈值 0.48、cls_id 为 7 的类别应用阈值 0.4；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>0.7</code>；</li>
</ul>
</td>
<td><code>0.7</code></td>
</tr>
</table>
3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测图像的输入路径。
    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`。
    - `boxes`: `(List[Dict])` 表示预测结果的类别id。
    - `labels`: `(List[str])` 表示预测结果的类别名称。
    - `cls_scores`: `(List[numpy.ndarray])` 表示属性预测结果的置信度。
    - `det_scores`: `(float)` 表示行人检测框的置信度。

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键为 `res` 对应的值是`Image.Image` 对象：一个用于显示属性识别结果的可视化图像。

此外，您可以获取行人属性识别产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config pedestrian_attribute_recognition --save_path ./my_path
```

若您获取了配置文件，即可对行人属性识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/pedestrian_attribute_recognition.yaml")

output = pipeline.predict(
    input="./pedestrian_attribute_002.jpg",
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改行人属性识别产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署

如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>
<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>获取行人属性识别结果。</p>
<p><code>POST /pedestrian-attribute-recognition</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>detThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>det_threshold</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>clsThreshold</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>cls_threshold</code> 参数说明。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pedestrians</code></td>
<td><code>array</code></td>
<td>行人的位置及属性等信息。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code> | <code>null</code></td>
<td>行人属性识别结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>pedestrians</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>行人位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。</td>
</tr>
<tr>
<td><code>attributes</code></td>
<td><code>array</code></td>
<td>行人属性。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>检测得分。</td>
</tr>
</tbody>
</table>
<p><code>attributes</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>属性标签。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>分类得分。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/pedestrian-attribute-recognition" # 服务URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected pedestrians:")
print(result["pedestrians"])
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果行人属性识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升行人属性识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于行人属性识别产线包含行人属性识别模块和行人检测模块，如果模型产线的效果不及预期，可能来自于其中任何一个模块。您可以对识别效果差的图片进行分析，进而确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。

<table>
<thead>
<tr>
<th>情形</th>
<th>微调模块</th>
<th>微调参考链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>行人检测不准</td>
<td>行人检测模块</td>
<td><a href="../../../module_usage/tutorials/cv_modules/human_detection.md">链接</a></td>
</tr>
<tr>
<td>属性识别不准</td>
<td>行人属性识别模块</td>
<td><a href="../../../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.md">链接</a></td>
</tr>
</tbody>
</table>

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
pipeline_name: pedestrian_attribute_recognition

SubModules:
  Detection:
    module_name: object_detection
    model_name: PP-YOLOE-L_human
    model_dir: null # 替换为微调后的行人检测模型权重路径
    batch_size: 1
    threshold: 0.5
  Classification:
    module_name: multilabel_classification
    model_name: PP-LCNet_x1_0_pedestrian_attribute
    model_dir: null # 替换为微调后的行人属性识别模型权重路径
    batch_size: 1
    threshold: 0.7
```
随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行行人属性识别产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline pedestrian_attribute_recognition \
        --input pedestrian_attribute_002.jpg \
        --device npu:0
```

当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用行人属性识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
