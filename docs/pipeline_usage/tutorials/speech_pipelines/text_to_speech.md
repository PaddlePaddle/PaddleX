---
comments: true
---

# 语音合成产线使用教程

## 1. 语音合成产线介绍
语音合成​​是一种前沿技术，能够将计算机生成的文本信息实时转换为自然流畅的人类语音信号。该技术已在智能助手、无障碍服务、导航播报、媒体娱乐等多个领域深度应用，显著提升人机交互体验，实现跨语言场景的高自然度语音输出。


> 推理耗时仅包含模型推理耗时，不包含前后处理耗时。

<p><b>语音合成模型：</b></p>
<table>
   <tr>
     <th >模型</th>
     <th >模型下载链接</th>
     <th >训练数据</th>
     <th>模型存储大小（MB）</th>
     <th >介绍</th>
   </tr>
   <tr>
     <td>fastspeech2_csmsc_pwgan_csmsc</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/fastspeech2_csmsc.tar">fastspeech2_csmsc</a><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/pwgan_csmsc.tar">pwgan_csmsc</a></td>
     <td >/</td>
     <td>768.1</td>
     <td rowspan="5">FastSpeech2 是微软开发的端到端文本转语音（TTS）模型，具备高效稳定的韵律控制能力。它采用非自回归架构，能实现快速高质量的语音合成，适用于虚拟助手、有声读物等多种场景。</td>
   </tr>
 </table>

## 2. 快速开始
PaddleX 支持在本地使用命令行或 Python 体验语音合成产线的效果。

在本地使用语音合成产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了 PaddleX 的 wheel 包安装。如果您希望选择性安装依赖，请参考安装教程中的相关说明。该产线对应的依赖分组为 `speech`。

### 2.1 本地体验

#### 2.1.1 命令行方式体验
一行命令即可快速体验文档图像预处理产线效果，使用 [示例文本](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline multilingual_speech_recognition \
        --input zh.wav 
```

相关的参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：

```plaintext
{'res': {'result': array([-8.118157e-04, ...,  6.217696e-05], shape=(38700,), dtype=float32)}}
```

运行结果参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的结果解释。

#### 2.1.2 Python脚本方式集成

上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="text_to_speech")

output = pipeline.predict(
    "今天天气真的很好"
)

for res in output:
    print(res)
    res.print()
    res.save_to_audio("./output/test.wav")
    res.save_to_json("./output")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化 text_to_speech 产线对象：具体参数说明如下：

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
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理插件。如果为 <code>None</code>，则使用配置文件或 <code>config</code> 中的配置。目前暂不支持。</td>
<td><code>bool</code> | <code>None</code></td>
<td>无</td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置。目前暂不支持。</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
</tr>
</tbody>
</table>

（2）调用 text_to_speech 产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<td>待预测数据</td>
<td><code>str</code></td>
<td>
<ul>
  <li><b>文件路径</b>，如语音文件的本地路径：<code>/root/data/text.txt</code></li>
  <li><b>合成的文字</b>，如<code>今天天气真不错</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为音频、保存为`json`文件的操作:

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
<td rowspan = "1"><code>save_to_audio()</code></td>
<td rowspan = "1">将结果保存为wav格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>


</table>

- 调用`print()` 方法会将结果打印到终端

- 调用`save_to_audio()` 方法会将上述内容保存到指定的`save_path`中

<!-- 此外，您可以获取 text_to_speech 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config multilingual_speech_recognition --save_path ./my_path
``` -->

若您获取了配置文件，即可对 text_to_speech 产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

例如，若您的配置文件保存在 `./my_path/text_to_speech.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/text_to_speech.yaml")
output = pipeline.predict(input="今天天气真的很好")
for res in output:
    res.print()
    res.save_to_json("./output/")
    res.save_to_audio("./output/test.wav")
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改 text_to_speech 产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署

如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<!-- <details><summary>API参考</summary>

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
<p>对音频进行多语种语音识别。</p>
<p><code>POST /multilingual-speech-recognition</code></p>
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
<td><code>audio</code></td>
<td><code>string</code></td>
<td>服务器可访问的音频文件的URL或路径。</td>
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
<td><code>text</code></td>
<td><code>string</code></td>
<td>语音识别结果文本。</td>
</tr>
<tr>
<td><code>segments</code></td>
<td><code>array</code></td>
<td>带时间戳的结果文本。</td>
</tr>
<tr>
<td><code>language</code></td>
<td><code>string</code></td>
<td>识别语种。</td>
</tr>
</tbody>
</table>
<p><code>segments</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td>语音片段ID。</td>
</tr>
<tr>
<td><code>seek</code></td>
<td><code>integer</code></td>
<td>语音片段指针。</td>
</tr>
<tr>
<td><code>start</code></td>
<td><code>number</code></td>
<td>语音片段开始时间。</td>
</tr>
<tr>
<td><code>end</code></td>
<td><code>number</code></td>
<td>语音片段结束时间。</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>语音片段识别文本。</td>
</tr>
<tr>
<td><code>tokens</code></td>
<td><code>array</code></td>
<td>语音片段的token id。</td>
</tr>
<tr>
<td><code>temperature</code></td>
<td><code>number</code></td>
<td>变速比例</td>
</tr>
<tr>
<td><code>avgLogProb</code></td>
<td><code>number</code></td>
<td>平均log概率。</td>
</tr>
<tr>
<td><code>compressionRatio</code></td>
<td><code>number</code></td>
<td>压缩比。</td>
</tr>
<tr>
<td><code>noSpeechProb</code></td>
<td><code>number</code></td>
<td>非语音概率。</td>
</tr>
</tbody>
</table>
</details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">
import base64
import requests

API_URL = "http://localhost:8080/multilingual-speech-recognition" # 服务URL
audio_path = "./zh.wav"

with open(audio_path, "rb") as file:
    audio_bytes = file.read()
    audio_data = base64.b64encode(audio_bytes).decode("ascii")

payload = {"audio": audio_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
print(result)
</code></pre></details>
</details>
<br/> -->

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/on_device_deployment.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。
