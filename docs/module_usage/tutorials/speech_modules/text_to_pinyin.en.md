---
comments: true
---

# Tutorial for Text To Pinyin Module

## I. Overview
Text to Pinyin is commonly used in the frontend of TTS to convert input Chinese text into a phonetic sequence with tones, providing pronunciation basis for subsequent acoustic models and audio generation.

## II. Supported Model List

<table>
  <tr>
    <th >Model</th>
    <th >Download link</th>
    <th >Model size</th>
    <th >Introduction</th>
  </tr>
  <tr>
    <td>G2PWModel</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/G2PWModel.tar">G2PWModel</a></td>
    <td>606M</td>
    <td rowspan="5"> g2pW is an open-source text to pinyin model, commonly used in the frontend of TTS. It converts input Chinese text into a tonal Pinyin sequence, providing pronunciation basis for subsequent acoustic models and audio generation</td>
  </tr>
</table>

## III. Quick Integration
Before quick integration, you need to install the PaddleX wheel package. For the installation method, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). After installing the wheel package, a few lines of code can complete the inference of the text to pinyin module. You can switch models under this module freely, and you can also integrate the model inference of the text to pinyin module into your project.


```python
from paddlex import create_model
model = create_model(model_name="G2PWModel")
output = model.predict(input="欢迎使用飞桨", batch_size=1)
for res in output:
    res.print()
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': '欢迎使用飞桨', 'result': ['huan1', 'ying2', 'shi3', 'yong4', 'fei1', 'jiang3']}}
```

The meanings of the runtime parameters are as follows:
- `input_path`: The storage path of the input text.
- `result`: Pinyin converted from the input text.

Related methods, parameters, and explanations are as follows:
* `create_model` for text to pinyin model, with specific explanations as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>The name of the model</td>
<td><code>str</code></td>
<td><code>G2PWModel</code></td>
<td><code>G2PWModel</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>The storage path of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the built-in model parameters of PaddleX are used by default. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the text to pinyin model is called for inference and prediction. The parameters of the `predict()` method are `input` and `batch_size`, with specific explanations as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted</td>
<td><code>str</code></td>
<td>
<ul>
  Input text, such as: <code>欢迎使用飞桨</code>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Currently only supports 1</td>
<td>1</td>
</tr>
</table>

* The prediction results are processed as `dict` type for each sample and support the operation of saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a file in <code>json</code> format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

* Additionally, the prediction results can also be obtained through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).
