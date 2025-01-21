---
comments: true
---

# 文档图像预处理产线使用教程

## 1. 文档图像预处理产线介绍

文档图像预处理产线集成了文档方向分类和形变矫正两大功能。文档方向分类可自动识别文档的四个方向（0°、90°、180°、270°），确保文档以正确的方向进行后续处理。几何形变矫正模型则用于修正文档拍摄或扫描过程中的几何扭曲，恢复文档的原始形状和比例。适用于数字化文档管理、OCR识别前处理、以及任何需要提高文档图像质量的场景。通过自动化的方向校正与形变矫正，该模块显著提升了文档处理的准确性和效率，为用户提供更为可靠的图像分析基础。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">

<b>通用文档图像预处理</b><b>产线中包含文档图像方向分类模块和文档图像矫正模块</b>。

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>3.84845</td>
<td>9.23735</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>


<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>CER </th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
<b>模型的精度指标测量自 [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html)。</b>

## 2. 快速开始
PaddleX 支持在本地使用命令行或 Python 体验文档图像预处理产线的效果。

在本地使用文档图像预处理产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验文档图像预处理产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline doc_preprocessor --input doc_test_rotated.jpg --device gpu:0 --save_path output
```
参数说明：

```
--pipeline：产线名称，此处为文档图像预处理产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
--save_path 预测结果保存路径
```

在执行上述 CLI 时，加载的是默认的文档图像预处理产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details><summary> 👉点击展开</summary>

<pre><code>paddlex --get_pipeline_config doc_preprocessor
</code></pre>
<p>执行后，文档图像预处理产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 <code>./my_path</code> ）：</p>
<pre><code>paddlex --get_pipeline_config doc_preprocessor --save_path ./my_path
</code></pre>
<p>获取产线配置文件后，可将 <code>--pipeline</code> 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 <code>./doc_preprocessor.yaml</code>，只需执行：</p>
<pre><code class="language-bash">paddlex --pipeline ./doc_preprocessor.yaml --input doc_test_rotated.jpg --device gpu:0
</code></pre>
<p>其中，<code>--model</code>、<code>--device</code> 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。</p></details>


运行后，得到的结果为：

<details><summary> 👉点击展开</summary>

<pre><code>{'res': {'input_path': 'doc_test_rotated.jpg', 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}}
</code></pre>
<p>其中，angle为文档方向分类结果。</p></details>

可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">

可视化图片默认不进行保存，您可以通过 `--save_path` 自定义保存路径，随后所有结果将被保存在指定路径下。

### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，文档图像预处理产线如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="doc_preprocessor")

output = pipeline.predict("doc_test_rotated.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

> ❗ Python脚本运行得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化 文档图像预处理产线对象：具体参数说明如下：

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
<td>无</td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线模型推理设备。支持：“gpu”，“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）调用文档图像预处理产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

<table>
<thead>
<tr>
<th>参数类型</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>支持直接传入Python变量，如numpy.ndarray表示的图像数据。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件路径，如图像文件的本地路径：<code>/root/data/img.jpg</code>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件URL，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg">示例</a>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code>。</td>
</tr>
<tr>
<td>dict</td>
<td>支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：<code>{\"img\": \"/root/data1\"}</code>。</td>
</tr>
<tr>
<td>list</td>
<td>支持传入列表，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]</code>。</td>
</tr>
</tbody>
</table>
（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

<table>
<thead>
<tr>
<th>方法</th>
<th>说明</th>
<th>方法参数</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>打印结果到终端</td>
<td><code>- format_json</code>：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br/><code>- indent</code>：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，仅当format_json为True时有效，默认为False；</td>
</tr>
<tr>
<td>save_to_json</td>
<td>将结果保存为json格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br/><code>- indent</code>：int类型，json格式化设置，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，默认为False；</td>
</tr>
<tr>
<td>save_to_img</td>
<td>将结果保存为图像格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；</td>
</tr>
</tbody>
</table>
若您获取了配置文件，即可对文档图像预处理产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/doc_preprocessor.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/doc_preprocessor.yaml")
output = pipeline.predict("doc_test_rotated.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
```


## 3. 开发集成/部署
如果文档图像预处理产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将文档图像预处理产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

下面是API参考和多语言服务调用示例：

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
<p>获取图像文档图像预处理结果。</p>
<p><code>POST /doc_preprocessor</code></p>
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
<td><code>file</code></td>
<td><code>string</code></td>
<td>服务可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。对于超过10页的PDF文件，只有前10页的内容会被使用。</td>
<td>是</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>文件类型。<code>0</code>表示PDF文件，<code>1</code>表示图像文件。若请求体无此属性，则服务将尝试根据URL自动推断文件类型。</td>
<td>否</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>推理参数。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p><code>inferenceParams</code>的属性如下：</p>
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
<td><code>use_doc_orientation_classify</code></td>
<td><code>bool</code></td>
<td>推理时，是否使用方向分类模块<code>use_doc_orientation_classify</code>为True，则将对图像进行方向调整，使其转为正常的文字阅读方向。</td>
<td>否</td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td><code>bool</code></td>
<td>推理时，是否使用矫正模块<code>use_doc_unwarping</code>为True，则将对图像进行矫正。</td>
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
<td><code>docPreprocessorResult</code></td>
<td><code>object</code></td>
<td>文档图像预处理结果。数组长度为1（对于图像输入）或文档页数与10中的较小者（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中每一页的处理结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>docPreprocessorResult</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>inputImage</code></td>
<td><code>string</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>warpImage</code></td>
<td><code>string</code></td>
<td>矫正结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>angle</code></td>
<td><code>int</code></td>
<td>角度分类结果。</td>
</tr>
</tbody>
</table>
</details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/doc-preprocessor&quot;
file_path = &quot;./demo.jpg&quot;

with open(file_path, &quot;rb&quot;) as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode(&quot;ascii&quot;)

payload = {&quot;file&quot;: file_data, &quot;fileType&quot;: 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
for i, res in enumerate(result[&quot;docPreprocessorResult&quot;]):
    print(&quot;Detected docwarps:&quot;)
    print(res[&quot;docwarps&quot;])
    layout_img_path = f&quot;layout_{i}.jpg&quot;
    with open(layout_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;layoutImage&quot;]))
    print(f&quot;Output image saved at {layout_img_path}&quot;)
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果文档图像预处理产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升文档图像预处理产线的在您的场景中的识别效果。

### 4.1 模型微调
由于文档图像预处理产线包含两个模块（文档图像方向分类和文档图像矫正），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对预处理效果差的图片进行分析，如果在分析过程中发现有较多的方向分类错误，那么可能是方向分类检测模型存在不足，您需要参考[文档图像方向分类模块开发教程](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md#四二次开发)章节，使用您的私有数据集对模型进行微调；文档图像矫正模型目前不支持二次开发。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径填写至产线配置文件中的 `model_dir` 即可：

```bash
......
  DocOrientationClassify:
    module_name: doc_text_orientation
    model_name: PP-LCNet_x1_0_doc_ori
    model_dir: ./output/best_model/inference # 修改此处为微调后模型的本地路径
......
```
随后， 参考[2. 快速开始](#2-快速开始)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行文档图像预处理产线的推理，使用的 Python 命令为：

```bash
paddlex --pipeline doc_preprocessor --input doc_test_rotated.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 `--device` 进行修改即可：

```bash
paddlex --pipeline doc_preprocessor --input doc_test_rotated.jpg --device npu:0
```
若您想在更多种类的硬件上使用文档图像预处理产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
