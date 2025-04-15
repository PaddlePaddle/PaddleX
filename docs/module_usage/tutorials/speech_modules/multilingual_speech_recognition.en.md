---
comments: true
---

# Multilingual Speech Recognition Module Tutorial

## I. Overview
Speech recognition is an advanced tool that can automatically convert human-spoken multiple languages into corresponding texts. This technology also plays an important role in various fields such as intelligent customer service, voice assistants, and meeting minutes. Multilingual speech recognition can support automatic language retrieval and recognize speech in multiple different languages.

## II. Supported Model List

### Whisper Model
Demo Link | Training Data | Size | Descriptions | CER | Model
:-----------: | :-----:| :-------: | :-----: | :-----: |:---------:|
Whisper | 680kh from internet | large: 5.8G,</br>medium: 2.9G,</br>small: 923M,</br>base: 277M,</br>tiny: 145M | Encoder:Transformer,</br> Decoder:Transformer, </br>Decoding method: </br>Greedy search | 0.027 </br>(large, Librispeech) | [whisper-large](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-large-model.tar.gz) </br>[whisper-medium](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-medium-model.tar.gz) </br>[whisper-small](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-small-model.tar.gz) </br>[whisper-base](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-base-model.tar.gz) </br>[whisper-tiny](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-tiny-model.tar.gz) </br>

## III. Quick Integration
Before quick integration, you need to install the PaddleX wheel package. For the installation method, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). After installing the wheel package, a few lines of code can complete the inference of the text recognition module. You can switch models under this module freely, and you can also integrate the model inference of the text recognition module into your project.

Before running the following code, please download the [demo audio](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav) to your local machine.

```python
from paddlex import create_model
model = create_model(model_name="whisper_large")
output = model.predict("./zh.wav", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': './zh.wav', 'result': {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 2.0, 'text': '我认为跑步最重要的就是', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 50464, 50464, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50564], 'temperature': 0, 'avg_logprob': -0.22779104113578796, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.026114309206604958}, {'id': 1, 'seek': 200, 'start': 2.0, 'end': 31.0, 'text': '给我带来了身体健康', 'tokens': [50364, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 51814], 'temperature': 0, 'avg_logprob': -0.21976988017559052, 'compression_ratio': 0.23684210526315788, 'no_speech_prob': 0.009023111313581467}], 'language': 'zh'}}}
```

The meanings of the runtime parameters are as follows:
- `input_path`: The storage path of the input audio file.
- `text`: The text result of speech recognition.
- `segments`: The text result with timestamps.
- `language`: The recognized language.

Related methods, parameters, and explanations are as follows:
* `create_model` for multilingual recognition model (here using `whisper_large` as an example), with specific explanations as follows:
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
<td><code>whisper_large, whisper_medium, whisper_base, whisper_small, whisper_tiny</code></td>
<td><code>whisper_large</code></td>
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
<td>The device used for model inference</td>
<td><code>str</code></td>
<td>It supports specifying specific GPU card numbers, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. </td>
<td><code>bool</code></td>
<td>None</td>
<td><code>False</code></td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the built-in model parameters of PaddleX are used by default. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the speech recognition model is called for inference and prediction. The parameters of the `predict()` method are `input` and `batch_size`, with specific explanations as follows:

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
  <li><b>File Path</b>, such as the local path of an audio file: <code>/root/data/audio.wav</code></li>
  <li><b>URL Link</b>, such as the network URL of an audio file: <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav">Example</a></li>
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

## IV. Custom Development
Currently, this model only supports inference.

### 4.1 Data Preparation
#### 4.1.1 Demo Data Download
You can use the following commands to download the Demo dataset to a specified folder:

```bash
wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 4.2 Model Training
Not support for now.

### **4.3 Model Evaluation**
Not support for now.

### **4.4 Model Inference and Model Integration**

#### 4.4.1 Model Inference
To perform inference prediction via the command line, simply use the following command:

Before running the following code, please download the [demo audio](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav) to your local machine.

```bash
python main.py -c paddlex/configs/modules/multilingual_speech_recognition/whisper_large.yaml \
    -o Global.mode=predict \
    -o Predict.input="./zh.wav"
```
the following steps are required for model inference:

* Specify the `.yaml` configuration file path for the model (here it is `whisper_large.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the input data path: `-o Predict.input="..."`
* Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).
* New Feature: Paddle 3.0 support CINN (Compiler Infrastructure for Neural Networks) to accelerate training speed when using GPU device. Please specify `-o Train.dy2st=True` to enable it.

#### 4.4.2 Model Integration
Models can be directly integrated into the PaddleX pipelines or into your own projects.

1.<b>Pipeline Integration</b>

No example for now.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the text recognition module. Refer to the [Quick Integration](#iii-quick-integration) Python example code. Simply replace the model with the path to your trained model.
