---
comments: true
---

# 开放词汇分割产线使用教程

## 1. 开放词汇分割产线介绍
开放词汇分割是一项图像分割任务，旨在根据文本描述、边框、关键点等除图像以外的信息作为提示，分割图像中对应的物体。它允许模型处理广泛的对象类别，而无需预定义的类别列表。这项技术结合了视觉和多模态技术，极大地提高了图像处理的灵活性和精度。开放词汇分割在计算机视觉领域具有重要应用价值，尤其在复杂场景下的对象分割任务中表现突出。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。本产线目前不支持对模型的二次开发，计划在后续支持。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg">

<b>通用开放词汇分割产线中包含了开放词汇分割模块，您可以根据下方的基准测试数据选择使用的模型</b>。

<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>

<p><b>通用图像开放词汇分割模块（可选）：</b></p>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>SAM-H_box</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_box_infer.tar">推理模型</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td rowspan="2">SAM（Segment Anything Model）是一种先进的图像分割模型，能够根据用户提供的简单提示（如点、框或文本）对图像中的任意对象进行分割。基于SA-1B数据集训练，有一千万的图像数据和十一亿掩码标注，在大部分场景均有较好的效果。其中SAM-H_box表示使用框作为分割提示输入，SAM会分割被框包裹主的主体；SAM-H_point表示使用点作为分割提示输入，SAM会分割点所在的主体。</td>
</tr>
<tr>
<td>SAM-H_point</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_point_infer.tar">推理模型</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
</tr>
</table>

<b>注：所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32</b>。

## 2. 快速开始

### 2.1 本地体验
> ❗ 在本地使用通用开放词汇分割产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.1.1 命令行方式体验
* 一行命令即可快速体验开放词汇分割产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline open_vocabulary_segmentation \
        --input open_vocabulary_segmentation.jpg \
        --prompt_type box \
        --prompt "[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]" \
        --save_path ./output \
        --device gpu:0
```
相关的参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：
```bash
{'res': {'input_path': 'open_vocabulary_segmentation.jpg', 'prompts': {'box_prompt': [[112.9, 118.4, 513.8, 382.1], [4.6, 263.6, 92.2, 336.6], [592.4, 260.9, 607.2, 294.2]]}, 'masks': '...', 'mask_infos': [{'label': 'box_prompt', 'prompt': [112.9, 118.4, 513.8, 382.1]}, {'label': 'box_prompt', 'prompt': [4.6, 263.6, 92.2, 336.6]}, {'label': 'box_prompt', 'prompt': [592.4, 260.9, 607.2, 294.2]}]}}
```
运行结果参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，其中开放词汇分割的可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg">

#### 2.1.2 Python脚本方式集成
* 上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="open_vocabulary_segmentation")
output = pipeline.predict(input="open_vocabulary_segmentation.jpg", prompt_type="box", prompt=[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]])
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化 开放词汇分割 产线对象，具体参数说明如下：

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
<td><code>None</code></td>
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
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

（2）调用 开放词汇分割 产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
  <li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
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
<td><code>prompt_type</code></td>
<td>模型推理时使用的提示类型</td>
<td><code>str</code></td>
<td>
<ul>
    <li><b>box</b>：使用边界框作为提示词输入, 如果设置为<code>box</code>, 输入的prompt需要是<code>list[list[float, float, float, float]]</code>的形式</li>
    <li><b>point</b>：使用点作为提示词输入, 如果设置为<code>point</code>, 输入的prompt需要是<code>list[list[float, float]]</code>的形式</li>
</ul>
</ul>
</td>
<td><code>无</code></td>
</tr>
<td><code>prompt</code></td>
<td>模型推理时具体使用的提示</td>
<td><code>list[list[float]]</code></td>
<td>
<ul>
    <li><b>list[list[float]]</b>：需要根据<code>prompt_type</code>的具体类型设置
</ul>
</ul>
</td>
<td><code>无</code></td>
</tr>

</table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">打印结果到终端</td>
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
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">将结果保存为json格式的文件</td>
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

    - `input_path`: `(str)` 待预测图像的输入路径

    - `prompts`: `(dict)` 该图片预测时使用的原始提示信息

    - `masks`: `...` 分割模型实际预测的mask，由于数据过大不便于直接print，因此用`...`替换，可以通过res.save_to_img将预测结果保存为图片，通过res.save_to_json将预测结果保存为json文件。

    - `mask_infos`: `(list)` 分割结果信息，对应`masks`中的元素，长度和`masks`相等，每个元素为一个字典，包含以下字段
      - `label`: `(str)` 对应的`masks`中元素由哪种类型的prompt预测获得, 如`box_prompt`表示对应的mask由边界框作为提示词获得
      - `prompt`: `list` 对应的`masks`中元素预测时具体使用的提示信息

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。

- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键为 `res`, 对应的值是一个 `Image.Image` 对象：一个用于显示 开放词汇分割 的预测结果。

此外，您可以获取 开放词汇分割 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config open_vocabulary_segmentation --save_path ./my_path
```

若您获取了配置文件，即可对开放词汇分割产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/open_vocabulary_segmentation.yaml")

output = pipeline.predict(
    input="./open_vocabulary_segmentation.jpg",
    prompt_type="box",
    prompt=[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改通用开放词汇分割产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.1.2 Python脚本方式](#212-python脚本方式集成)中的示例代码。

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
<p>对图像进行目标分割。</p>
<p><code>POST /open-vocabulary-segmentation</code></p>
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
<td><code>prompt</code></td>
<td><code>array</code></td>
<td>预测使用的提示。</td>
<td>是</td>
</tr>
<tr>
<td><code>promptType</code></td>
<td><code>string</code></td>
<td>预测使用的提示类型。</td>
<td>是</td>
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
<td><code>masks</code></td>
<td><code>array</code></td>
<td>分割的预测结果。</td>
</tr>
<tr>
<td><code>maskInfos</code></td>
<td><code>array</code></td>
<td>和masks字段中的元素一一对应，记录masks中对应分割结果所使用的对应prompt。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>分割结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<b>注意</b>：考虑到网络传输, masks字段中记录的分割结果经过<code>rle</code>编码结果, 实际使用时需要使用<code>pycocotools.mask.decode</code>做对应的解码即可获得原始的分割结果。
</br>
<p><code>detectedObjects</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td>生成mask所使用的prompt类别。</td>
</tr>
<tr>
<td><code>prompt</code></td>
<td><code>array</code></td>
<td>prompt数组。</td>
</tr>
</tbody>
</table>

<p><code>result</code>示例如下：</p>
<pre><code class="language-python">
{
    'masks': [rle_mask1, rle_mask2, rle_mask3]
    'mask_infos': [
        {'label': 'box_prompt', 'prompt': [112.9, 118.4, 513.8, 382.1]},
        {'label': 'box_prompt', 'prompt': [4.6, 263.6, 92.2, 336.6]},
        {'label': 'box_prompt', 'prompt': [592.4, 260.9, 607.2, 294.2]}
    ]
}
</code>
</details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/open-vocabulary-segmentation" # 服务URL
image_path = "./open_vocabulary_segmentation.jpg"
output_image_path = "./out.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "image": image_data, # Base64编码的文件内容或者图像URL
    "prompt_type": "box",
    "prompt": [[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]
}

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
image_base64 = result["image"]
image = base64.b64decode(image_base64)
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nresult(with rle encoded binary mask):")
print(result)
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
当前产线暂时不支持微调训练，仅支持推理集成。关于该产线的微调训练，计划在未来支持。

## 5. 多硬件支持
当前产线暂时仅支持GPU和CPU推理。关于该产线对于更多硬件的适配，计划在未来支持。
