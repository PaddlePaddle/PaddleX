---
comments: true
---

# Text to Speech pipeline Tutorial

## 1. Introduction to Text to Speech pipeline
Text to Speech is a cutting-edge technology capable of converting computer-generated text information into natural and fluent human speech signals in real-time. This technology has been deeply applied across multiple domains including virtual assistants, accessibility services, navigation announcements, and media entertainment, significantly enhancing human-computer interaction experiences and enabling highly natural voice output in cross-linguistic scenarios.

<p><b>Text to Speech Model:</b></p>
<table>
   <tr>
     <th >Model</th>
     <th >Model Download Link</th>
     <th >Training Data</th>
     <th>Model Storage Size (MB)</th>
     <th >Introduction</th>
   </tr>
   <tr>
     <td>fastspeech2_csmsc_pwgan_csmsc</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/fastspeech2_csmsc.tar">fastspeech2_csmsc</a><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/pwgan_csmsc.tar">pwgan_csmsc</a></td>
     <td >/</td>
     <td>768.1</td>
     <td rowspan="5">FastSpeech2 is an end-to-end text-to-speech (TTS) model developed by Microsoft, featuring efficient and stable prosody control capabilities. Utilizing a non-autoregressive architecture, it enables fast and high-quality speech synthesis, making it suitable for various applications such as virtual assistants and audiobook production.</td>
   </tr>
 </table>

## 2. Quick Start
PaddleX supports experiencing the multilingual speech recognition pipeline locally using the command line or Python.

Before using the multilingual speech recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `speech`.

### 2.1 Local Experience

#### 2.1.1 Command Line Experience
PaddleX supports experiencing the text to speech pipeline locally using the command line or Python.

```bash
paddlex --pipeline text_to_speech \
        --input "今天天气真的很好"
```

The relevant parameter descriptions can be found in the parameter descriptions in [2.1.2 Integration via Python Script]().

After running, the result will be printed to the terminal, as follows:

```plaintext
{'res': {'result': array([-8.118157e-04, ...,  6.217696e-05], shape=(38700,), dtype=float32)}}
```

The explanation of the result parameters can refer to the result explanation in [2.1.2 Integration with Python Script](#212-integration-with-python-script).。

#### 2.1.2 Integration with Python Script

The above command line is for quickly experiencing and viewing the effect. Generally speaking, in a project, it is often necessary to integrate through code. You can complete the rapid inference of the pipeline with just a few lines of code. The inference code is as follows:

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

In the above Python script, the following steps are executed:

（1）The <code>text to speech</code> pipeline object is instantiated through <code>create_pipeline()</code>. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline. It supports specifying the specific card number of the GPU, such as "gpu:0", the specific card number of other hardware, such as "npu:0"</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin. If set to <code>None</code>, the setting from the configuration file or <code>config</code> will be used. Not supported for now.</td>
<td><code>bool</code> | <code>None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration. Not supported for now.</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
</tr>
</tbody>
</table>

（2）The <code>predict()</code> method of the <code>text to speech</code> pipeline object is called to perform inference and prediction. This method will return a <code>generator</code>. Below are the parameters and their descriptions for the <code>predict()</code> method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted</td>
<td><code>str</code></td>
<td>
<ul>
  <li><b>File path</b>, such as the local path of an text file:<code>/root/data/text.txt</code></li>
  <li><b>Text to be synthesized</b>, such as<code>今天天气真不错</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

（3）Process the prediction results. The prediction result for each sample is of the AudioResult type and supports operations such as printing, saving as an audio, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Descrition</th>
<th>Parameter</th>
<th>Parameter type</th>
<th>Parameter Description</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td><td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. When it is a directory, the saved file name is consistent with the input file type naming</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>

<tr>
<td rowspan = "1"><code>save_to_audio()</code></td>
<td rowspan = "1">Save the result as a wav file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path. When it is a directory, the saved file name is consistent with the input file type name.</td>
<td>None</td>
</tr>


</table>

- Calling the `print()` method will print the result to the terminal, with the printed content explained as follows:

- Calling the `save_to_audio()` method will save the above content to the specified `save_path`.

<!-- 此外，您可以获取 text_to_speech 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config multilingual_speech_recognition --save_path ./my_path
``` -->

Once you have the configuration file, you can customize the text_to_speech pipeline configuration by modifying the `pipeline` parameter in the `create_pipeline` method to the path to the pipeline configuration file. An example is as follows:

For example, if your configuration file is saved at `./my_path/text_to_speech.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/text_to_speech.yaml")
output = pipeline.predict(input="今天天气真的很好")
for res in output:
    res.print()
    res.save_to_json("./output/")
    res.save_to_audio("./output/test.wav")
```

<b>Note:</b> The parameters in the configuration file are the initialization parameters for the pipeline. If you want to change the initialization parameters of the <code>text to speech</code> pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path of the configuration file with <code>--pipeline</code>.

## 3. Development Integration/Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can directly proceed with development integration/deployment.

If you need to apply the pipeline directly in your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure the efficient operation of the system and the smoothness of the user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing to achieve significant acceleration of the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving Deployment</b>: Serving Deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports multiple pipeline serving deployment solutions. For detailed pipeline serving deployment procedures, please refer to the [PaddleX Serving Deployment Guide](../../../pipeline_deploy/serving.en.md).

📱 <b>On-Device Deployment</b>: Edge deployment is a method that places computational and data processing capabilities directly on user devices, allowing them to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed procedures, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.en.md).
