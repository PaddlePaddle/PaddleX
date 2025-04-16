---
comments: true
---

# 多语种语音识别模块使用教程

## 一、概述
语音识别是一种先进的工具，它能够将人类口述的多种语言自动转换为相应的文本或命令。该技术还在智能客服、语音助手、会议记录等多个领域发挥着重要作用。多语种语音识别则可以支持自动进行语种检索，支持多种不同语言的语音的识别。

## 二、支持模型列表

### Whisper Model
<table>
  <tr>
    <th >模型</th>
    <th >模型下载链接</th>
    <th >训练数据</th>
    <th >模型大小</th>
    <th >词错率</th>
    <th >介绍</th>
  </tr>
  <tr>
    <td>whisper_large</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/whisper_large.tar">whisper_large</a></td>
    <td >680kh</td>
    <td>5.8G</td>
    <td>2.7 (Librispeech)</td>
    <td rowspan="5">Whisper 是 OpenAI 开发的多语言自动语音识别模型，具备高精度和鲁棒性。它采用端到端架构，能处理嘈杂环境音频，适用于语音助理、实时字幕等多种应用。</td>
  </tr>
  <tr>
    <td>whisper_medium</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/whisper_medium.tar">whisper_medium</a></td>
    <td>680kh</td>
    <td>2.9G</td>
    <td>-</td>
  </tr>
  <tr>
    <td>whisper_small</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/whisper_small.tar">whisper_small</a></td>
    <td>680kh</td>
    <td>923M</td>
    <td>-</td>
  </tr>
  <tr>
    <td>whisper_base</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/whisper_base.tar">whisper_base</a></td>
    <td>680kh</td>
    <td>277M</td>
    <td>-</td>
  </tr>
  <tr>
    <td>whisper_small</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/whisper_tiny.tar">whisper_tiny</a></td>
    <td>680kh</td>
    <td>145M</td>
    <td>-</td>
  </tr>
</table>

## 三、快速集成
在快速集成前，首先需要安装 PaddleX 的 wheel 包，wheel的安装方式请参考[PaddleX本地安装教程](../../../installation/installation.md)。完成 wheel 包的安装后，几行代码即可完成多语种语音识别模块的推理，可以任意切换该模块下的模型，您也可以将多语种语音识别模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例语音](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav)到本地。

```python
from paddlex import create_model
model = create_model(model_name="whisper_large")
output = model.predict(input="./zh.wav", batch_size=1)
for res in output:
    res.print()
    res.save_to_json(save_path="./output/res.json")
```
运行后，得到的结果为：
```bash
{'res': {'input_path': './zh.wav', 'result': {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 2.0, 'text': '我认为跑步最重要的就是', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 50464, 50464, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50564], 'temperature': 0, 'avg_logprob': -0.22779104113578796, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.026114309206604958}, {'id': 1, 'seek': 200, 'start': 2.0, 'end': 31.0, 'text': '给我带来了身体健康', 'tokens': [50364, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 51814], 'temperature': 0, 'avg_logprob': -0.21976988017559052, 'compression_ratio': 0.23684210526315788, 'no_speech_prob': 0.009023111313581467}], 'language': 'zh'}}}
```
运行结果参数含义如下：
- `input_path`: 输入音频存放路径
- `result`: 识别结果
    -  `text`: 语音识别结果文本
    -  `segments`: 带时间戳的结果文本
        * `id`: ID
        * `seek`: 语音片段指针
        * `start`: 片段开始时间
        * `end`: 片段结束时间
        * `text`: 片段识别文本
        * `tokens`: 片段文本的 token id
        * `temperature`: 变速比例
        * `avg_logprob`: 平均 log 概率
        * `compression_ratio`: 压缩比
        * `no_speech_prob`: 非语音概率
    - `language`: 识别语种

相关方法、参数等说明如下：
* `create_model`多语种识别模型（此处以`whisper_large`为例），具体说明如下：
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
<td><code>model_name</code></td>
<td>模型名称</td>
<td><code>str</code></td>
<td><code>whisper_large、whisper_medium、whisper_base、whisper_small、whisper_tiny</code></td>
<td><code>whisper_large</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>device</code></td>
<td>模型推理设备</td>
<td><code>str</code></td>
<td>支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用多语种语音识别模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` 和 `batch_size`，具体说明如下：

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
  <li><b>文件路径</b>，如语音文件的本地路径：<code>/root/data/audio.wav</code></li>
  <li><b>URL链接</b>，如语音文件的网络URL：<a href = "https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav">示例</a></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小</td>
<td><code>int</code></td>
<td>目前仅支持1</td>
<td>1</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，支持保存为`json`文件的操作:

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
</table>

* 此外，也支持通过属性获取预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的<code>json</code>格式的结果</td>
</tr>

</table>

关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。
