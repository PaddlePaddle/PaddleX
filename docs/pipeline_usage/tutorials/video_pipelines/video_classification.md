---
comments: true
---

# 通用视频分类产线使用教程

## 1. 通用视频分类产线介绍
视频分类是一种将视频片段分配到预定义类别的技术。它广泛应用于动作识别、事件检测和内容推荐等领域。视频分类可以识别各种动态事件和场景，如体育活动、自然现象、交通状况等，并根据其特征将其归类。通过使用深度学习模型，尤其是卷积神经网络（CNN）和循环神经网络（RNN）的结合，视频分类能够自动提取视频中的时空特征并进行准确分类。这种技术在视频监控、媒体检索和个性化推荐系统中具有重要应用.

通用视频分类产线用于解决视频分类任务，提取视频中的主题和类别信息以标签形式输出，本产线是一个集成了业界知名的 PP-TSM 和 PP-TSMv2的视频分类系统，支持400种视频类别的识别。基于本产线，可实现视频内容的精准分类，使用场景覆盖媒体、安防、教育、交通等各个领域。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/01.jpg">
<b>通用视频分类产线中包含了视频分类模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。



<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top1 Acc(%)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-TSM-R50_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSM-R50_8frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSM-R50_8frames_uniform_pretrained.pdparams">训练模型</a></td>
<td>74.36</td>
<td>93.4 M</td>
<td rowspan="1">
PP-TSM是一种百度飞桨视觉团队自研的视频分类模型。该模型基于ResNet-50骨干网络进行优化，从数据增强、网络结构微调、训练策略、BN层优化、预训练模型选择、模型蒸馏等6个方面进行模型调优，在中心采样评估方式下，Kinetics-400上精度较原论文实现提升3.95个点
</td>
</tr>

<tr>
<td>PP-TSMv2-LCNetV2_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_8frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_8frames_uniform_pretrained.pdparams">训练模型</a></td>
<td>71.71</td>
<td>22.5 M</td>
<td rowspan="2">PP-TSMv2是轻量化的视频分类模型，基于CPU端模型PP-LCNetV2进行优化，从骨干网络与预训练模型选择、数据增强、tsm模块调优、输入帧数优化、解码速度优化、DML蒸馏、LTA模块等7个方面进行模型调优，在中心采样评估方式下，精度达到75.16%，输入10s视频在CPU端的推理速度仅需456ms。</td>
</tr>
<tr>
<td>PP-TSMv2-LCNetV2_16frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_16frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_16frames_uniform_pretrained.pdparams">训练模型</a></td>
<td>73.11</td>
<td>22.5 M</td>
</tr>

</table>

<p><b>注：以上精度指标为 <a href="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md">K400</a> 验证集 Top1 Acc。</b></p></details>

## 2. 快速开始

PaddleX 支持在本地使用命令行或 Python 体验产线的效果。

在本地使用通用视频分类产线前，请确保您已经按照PaddleX本地安装教程完成了PaddleX的wheel包安装。

#### 2.1 命令行方式体验
一行命令即可快速体验视频分类产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline video_classification \
    --input general_video_classification_001.mp4 \
    --topk 5 \
    --save_path ./output \
    --device gpu:0
```
相关的参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：
```bash
{'res': {'input_path': 'general_video_classification_001.mp4', 'class_ids': array([  0, ..., 162], dtype=int32), 'scores': [0.91997, 0.07052, 0.00237, 0.00214, 0.00158], 'label_names': ['abseiling', 'rock_climbing', 'climbing_tree', 'riding_mule', 'ice_climbing']}}
```
运行结果参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的结果解释。


可视化结果保存在`save_path`下，其中视频分类的可视化结果如下：
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/02.jpg" style="width: 70%">


#### 2.2 Python脚本方式集成
* 上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="video_classification")

output = pipeline.predict("general_video_classification_001.mp4", topk=5)
for res in output:
    res.print()
    res.save_to_video(save_path="./output/")
    res.save_to_json(save_path="./output/")
```


在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化视频分类产线对象，具体参数说明如下：

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

（2）调用 通用视频分类产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<tbody>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填</td>
<td><code>str|list</code></td>
<td>
<ul>
  <li><b>str</b>：如视频文件的本地路径：<code>/root/data/video.mp4</code>；<b>如URL链接</b>，如视频文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4">示例</a>；<b>如本地目录</b>，该目录下需包含待预测视频，如本地路径：<code>/root/data/</code></li>
  <li><b>List</b>：列表元素需为上述类型数据，如 <code>[\"/root/data/video1.mp4\", \"/root/data/video2.mp4\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<td><code> topk</code></td>
<td>预测结果的前 <code>topk</code> 个类别和对应的分类概率</td>
<td><code>int|None</code></td>
<td>
<ul>
    <li><b>int</b>：大于 <code>0</code> 的任意整数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>1</code>。</td>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>

</tbody>
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
<td><code>save_to_video()</code></td>
<td>将结果保存为视频格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测视频的输入路径
    - `class_ids`: `(numpy.ndarray)` 视频分类的id 列表
    - `scores`: `(List[float])` 视频分类的置信度分数列表
    - `label_names`: `(List[str])` 视频分类的类别列表


- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_video_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_video()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_video_basename}_res.{your_video_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果视频，不建议直接指定为具体的文件路径，否则多个视频会被覆盖，仅保留最后一个视频)

* 此外，也支持通过属性获取带结果的可视化视频和预测结果，具体如下：


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
<td rowspan = "2"><code>video</code></td>
<td rowspan = "2">获取格式为 <code>dict</code> 的可视化视频</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `video` 属性返回的预测结果是一个字典类型的数据。其中，键为 `res`，对应的值是一个元组，元组的第一个值是可视化视频数组，维度是（视频帧数，视频高度，视频宽度，视频通道数）；第二个值是帧率。

此外，您可以获取视频分类产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config video_classification --save_path ./my_path
```

若您获取了配置文件，即可对视频分类产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/video_classification.yaml")

output = pipeline.predict("general_video_classification_001.mp4", topk=5)
for res in output:
    res.print()
    res.save_to_video(save_path="./output/")
    res.save_to_json(save_path="./output/")
```
<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改通用视频分类产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

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
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
</tbody>
</table>
<p>响应体还可能有<code>result</code>属性，类型为<code>object</code>，其中存储操作结果信息。</p>
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
<p>对视频进行分类。</p>
<p><code>POST /video-classification</code></p>
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
<td><code>video</code></td>
<td><code>string</code></td>
<td>服务器可访问的视频文件的URL或视频文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>topk</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>topk</code> 参数说明。</td>
<td>否</td>
</tr>
</tbody>
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
<td><code>categories</code></td>
<td><code>array</code></td>
<td>视频类别信息。</td>
</tr>
</tbody>
</table>
<p><code>categories</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>id</code></td>
<td><code>integer</code></td>
<td>类别ID。</td>
</tr>
<tr>
<td><code>name</code></td>
<td><code>string</code></td>
<td>类别名称。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>类别得分。</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
&quot;categories&quot;: [
{
&quot;id&quot;: 5,
&quot;name&quot;: &quot;兔子&quot;,
&quot;score&quot;: 0.93
}
],
&quot;video&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/video-classification&quot; # 服务URL
video_path = &quot;./demo.mp4&quot;
output_video_path = &quot;./out.mp4&quot;

# 对本地视频进行Base64编码
with open(video_path, &quot;rb&quot;) as file:
    video_bytes = file.read()
    video_data = base64.b64encode(video_bytes).decode(&quot;ascii&quot;)

payload = {&quot;video&quot;: video_data}  # Base64编码的文件内容或者视频URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
print(&quot;Categories:&quot;)
print(result[&quot;categories&quot;])
</code></pre></details>
<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string videoPath = &quot;./demo.mp4&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // 对本地视频进行Base64编码
    std::ifstream file(videoPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;video&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post(&quot;/video-classification&quot;, headers, body, &quot;application/json&quot;);
    // 处理接口返回数据
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];
        auto categories = result[&quot;categories&quot;];
        std::cout &lt;&lt; &quot;\nCategories:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : categories) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = &quot;http://localhost:8080/video-classification&quot;; // 服务URL
        String videoPath = &quot;./demo.mp4&quot;; // 本地视频

        // 对本地视频进行Base64编码
        File file = new File(videoPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String videoData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;video&quot;, videoData); // Base64编码的文件内容或者视频URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                JsonNode categories = result.get(&quot;categories&quot;);
                System.out.println(&quot;\nCategories: &quot; + categories.toString());
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/video-classification&quot;
    videoPath := &quot;./demo.mp4&quot;

    // 对本地视频进行Base64编码
    videoBytes, err := ioutil.ReadFile(videoPath)
    if err != nil {
        fmt.Println(&quot;Error reading video file:&quot;, err)
        return
    }
    videoData := base64.StdEncoding.EncodeToString(videoBytes)

    payload := map[string]string{&quot;video&quot;: videoData} // Base64编码的文件内容或者视频URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    // 调用API
    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    // 处理接口返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Categories []map[string]interface{} `json:&quot;categories&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    fmt.Println(&quot;\nCategories:&quot;)
    for _, category := range respData.Result.Categories {
        fmt.Println(category)
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = &quot;http://localhost:8080/video-classification&quot;;
    static readonly string videoPath = &quot;./demo.mp4&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地视频进行Base64编码
        byte[] videoBytes = File.ReadAllBytes(videoPath);
        string video_data = Convert.ToBase64String(videoBytes);

        var payload = new JObject{ { &quot;video&quot;, video_data } }; // Base64编码的文件内容或者视频URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        Console.WriteLine(&quot;\nCategories:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;categories&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/video-classification'
const videoPath = './demo.mp4'

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'video': encodeImageToBase64(videoPath)  // Base64编码的文件内容或者视频URL
  })
};

// 对本地视频进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) =&gt; {
    // 处理接口返回数据
    const result = response.data[&quot;result&quot;];
    console.log(&quot;\nCategories:&quot;);
    console.log(result[&quot;categories&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/video-classification&quot;; // 服务URL
$video_path = &quot;./demo.mp4&quot;;

// 对本地视频进行Base64编码
$video_data = base64_encode(file_get_contents($video_path));
$payload = array(&quot;video&quot; =&gt; $video_data); // Base64编码的文件内容或者视频URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)[&quot;result&quot;];
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用视频分类产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用视频分类产线的在您的场景中的识别效果。

### 4.1 模型微调

由于通用视频分类产线仅包含视频分类模块，模型产线的效果如果不及预期。您可以对识别效果差的视频进行分析，并参考以下表格中对应的微调教程链接进行模型微调。


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
      <td>视频分类都不准</td>
      <td>视频分类模块</td>
      <td><a href="../../../module_usage/tutorials/video_modules/video_classification.md">链接</a></td>
    </tr>

  </tbody>
</table>

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
...
SubModules:
  VideoClassification:
    module_name: video_classification
    model_name: PP-TSMv2-LCNetV2_8frames_uniform
    model_dir: null # 替换为微调后的视频分类模型权重路径
    batch_size: 1
    topk: 1

...
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行 视频分类产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline video_classification \
    --input general_video_classification_001.mp4 \
    --topk 5 \
    --save_path ./output \
    --device npu:0
```

当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用通用视频分类产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
