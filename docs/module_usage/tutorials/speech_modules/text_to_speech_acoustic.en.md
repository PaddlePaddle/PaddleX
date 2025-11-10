---
comments: true
---

# 语音合成声学模型使用教程

## I. Overview
The acoustic model for speech synthesis is the core component of speech synthesis technology. Its key characteristic lies in utilizing deep learning and other techniques to transform text into lifelike voice output while enabling fine-grained control over features such as speech rate and prosody. It is primarily applied in fields such as intelligent voice assistants, navigation announcements, and film and television dubbing.

## Supported Model List

### Fastspeech Model
<table>
  <tr>
    <th >Model</th>
    <th >Download link</th>
    <th >training data</th>
    <th >Introduction</th>
    <th >Model Storage Size (MB)</th>
  </tr>
  <tr>
    <td>fastspeech2_csmsc</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/fastspeech2_csmsc.tar">fastspeech2_csmsc</a></td>
    <td>\</td>
    <td>157M</td>
    <td rowspan="1">FastSpeech2 is an end-to-end text-to-speech (TTS) model developed by Microsoft, featuring efficient and stable prosody control. It adopts a non-autoregressive architecture that enables fast and high-quality speech synthesis, suitable for various scenarios such as virtual assistants and audiobooks.</td>
  </tr>
</table>

## 3. Quick Integration
Before quick integration, first install the PaddleX wheel package. For wheel installation methods, please refer to [PaddleX Local Installation Tutorial](../../../installation/installation.md). After installing the wheel package, inference for the multilingual speech synthesis acoustic module can be completed with just a few lines of code. You can freely switch models within this module, or integrate model inference from the multilingual speech synthesis module into your project.
<!-- Before running the following code, please download the [sample audio](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav){target="_blank"} to your local machine. -->

```python
from paddlex import create_model
model = create_model(model_name="fastspeech2_csmsc")
output = model.predict(input=[151, 120, 182, 82, 182, 82, 174, 75, 262, 51, 37, 186, 38, 233]. , batch_size=1)
for res in output:
    res.print()
    res.save_to_json(save_path="./output/res.json")
```
After running, the results are:
```bash
{'result': array([[-2.96321  , ..., -4.552117 ],
  ...,
  [-2.0465052, ..., -3.695221 ]], dtype=float32)}
```

The meanings of the running result parameters are as follows:

- `input_path`: Input audio storage path
- `result`: Output mel spectrogram result


Explanations of related methods and parameters are as follows:

* `create_model`multilingual recognition model (using `fastspeech2_csmsc` as an example), details are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>The name of the model</td>
<td><code>str</code></td>
<td><code>fastspeech2_csmsc</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>The storage path of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Model inference device</td>
<td><code>str</code></td>
<td>Supports specifying specific GPU card numbers (e.g. "gpu:0"), other hardware card numbers (e.g. "npu:0"), or CPU (e.g. "cpu")</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference plugin. Currently not supported.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration. Currently not supported.</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</table>

* Among these, `model_name` must be specified. After specifying `model_name`, PaddleX's built-in model parameters are used by default. When `model_dir` is specified, the user-defined model is used.
* Call the multilingual speech synthesis model's `predict()` method for inference prediction. The `predict()` method parameters include `input` and `batch_size`, with details as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted</td>
<td><code>str</code></td>
<td>
Currently only supports tensor-type input_phone_ids, such as [151, 120, 182, 82, 182, 82, 174, 75, 262, 51, 37, 186], etc.
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>batch size</td>
<td><code>int</code></td>
<td>Currently only 1 is supported</td>
<td>1</td>
</tr>
</table>

* Process the prediction results. The prediction result for each sample is a corresponding Result object, which supports saving as `json` files:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify output <code>JSON</code> data for better readability. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters to Unicode. When set to <code>True</code>, all non-ASCII characters will be escaped; <code>False</code> preserves original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save results as json file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>File save path. When it is a directory, the file name follows the input file type naming convention</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify output <code>JSON</code> data for better readability. Only effective when <code>format_json</code> is <code>True</code>/td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters to Unicode. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> preserves original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

* Additionally, the prediction results can also be obtained through attributes:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get prediction results in <code>json</code> format</td>
</tr>

</table>

For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).
