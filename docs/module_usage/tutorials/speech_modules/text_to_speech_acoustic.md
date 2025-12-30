---
comments: true
---

# 语音合成声学模型使用教程

## 一、概述
语音合成声学模型是语音合成技术的核心组件，其特点在于通过深度学习等技术将文本转化为逼真的语音输出，并支持细粒度控制语速、韵律等特征。主要应用于智能语音助手、导航播报、影视配音等领域。

## 二、支持模型列表

### Fastspeech Model
<table>
  <tr>
    <th >模型</th>
    <th >模型下载链接</th>
    <th >训练数据</th>
    <th>模型存储大小（MB）</th>
    <th >介绍</th>
  </tr>
  <tr>
    <td>fastspeech2_csmsc</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/fastspeech2_csmsc.tar">fastspeech2_csmsc</a></td>
    <td>\</td>
    <td>157M</td>
    <td rowspan="1">FastSpeech2 是微软开发的端到端文本转语音（TTS）模型，具备高效稳定的韵律控制能力。它采用非自回归架构，能实现快速高质量的语音合成，适用于虚拟助手、有声读物等多种场景。</td>
  </tr>
</table>

## 三、快速集成
在快速集成前，首先需要安装 PaddleX 的 wheel 包，wheel的安装方式请参考[PaddleX本地安装教程](../../../installation/installation.md)。完成 wheel 包的安装后，几行代码即可完成多语种语音合成声学模块的推理，可以任意切换该模块下的模型，您也可以将多语种语音合成模块中的模型推理集成到您的项目中。
<!-- 运行以下代码前，请您下载[示例语音](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav)到本地。 -->

```python
from paddlex import create_model
model = create_model(model_name="fastspeech2_csmsc")
output = model.predict(input=[151, 120, 182, 82, 182, 82, 174, 75, 262, 51, 37, 186, 38, 233]. , batch_size=1)
for res in output:
    res.print()
    res.save_to_json(save_path="./output/res.json")
```
运行后，得到的结果为：
```bash
{'result': array([[-2.96321  , ..., -4.552117 ],
  ...,
  [-2.0465052, ..., -3.695221 ]], dtype=float32)}
```
运行结果参数含义如下：
- `input_path`: 输入音频存放路径
- `result`: 输出mel谱结果

相关方法、参数等说明如下：
* `create_model`多语种识别模型（此处以`fastspeech2_csmsc`为例），具体说明如下：
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
<td><code>fastspeech2_csmsc</code></td>
<td><code>无</code></td>
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
<td>是否启用高性能推理插件。目前暂不支持。</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置。目前暂不支持。</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
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
  输入的input_phone_ids, 目前只支持tensor类型，如[151, 120, 182, 82, 182, 82, 174, 75, 262, 51, 37, 186]等
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
