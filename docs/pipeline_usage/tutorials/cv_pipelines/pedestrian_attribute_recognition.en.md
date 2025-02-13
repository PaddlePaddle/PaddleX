---
comments: true
---

# Pedestrian Attribute Recognition Pipeline Tutorial

## 1. Introduction to Pedestrian Attribute Recognition Pipeline
Pedestrian attribute recognition is a key function in computer vision systems, used to locate and label specific characteristics of pedestrians in images or videos, such as gender, age, clothing color, and style. This task not only requires accurately detecting pedestrians but also identifying detailed attribute information for each pedestrian. The pedestrian attribute recognition pipeline is an end-to-end serial system for locating and recognizing pedestrian attributes, widely used in smart cities, security surveillance, and other fields, significantly enhancing the system's intelligence level and management efficiency.This pipeline also offers a flexible service-oriented deployment approach, supporting the use of multiple programming languages on various hardware platforms. Moreover, this pipeline provides the capability for secondary development. You can train and optimize models on your own dataset based on this pipeline, and the trained models can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/01.jpg"/>
<b>The pedestrian attribute recognition pipeline includes a pedestrian detection module and a pedestrian attribute recognition module</b>, with several models in each module. Which models to use specifically can be selected based on the benchmark data below. <b>If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model storage size, choose models with smaller storage</b>.


<p><b>Pedestrian Detection Module</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE-L_human</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_human_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_human_pretrained.pdparams">Trained Model</a></td>
<td>48.0</td>
<td>81.9</td>
<td>33.27 / 9.19</td>
<td>173.72 / 173.72</td>
<td>196.02</td>
<td rowspan="2">Pedestrian detection model based on PP-YOLOE</td>
</tr>
<tr>
<td>PP-YOLOE-S_human</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_human_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_human_pretrained.pdparams">Trained Model</a></td>
<td>42.5</td>
<td>77.9</td>
<td>9.94 / 3.42</td>
<td>54.48 / 46.52</td>
<td>28.79</td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are mAP(0.5:0.95) on the CrowdHuman dataset. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Pedestrian Attribute Recognition Module</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_pedestrian_attribute</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams">Trained Model</a></td>
<td>92.2</td>
<td>2.35 / 0.49</td>
<td>3.17 / 1.25</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_pedestrian_attribute is a lightweight pedestrian attribute recognition model based on PP-LCNet, covering 26 categories.</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are mA on PaddleX's internally built dataset. GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start

All model pipelines provided by PaddleX can be quickly experienced. You can experience the effect of the pedestrian attribute recognition pipeline on the community platform, or you can use the command line or Python locally to experience the effect of the pedestrian attribute recognition pipeline.

### 2.1 Online Experience

You can [experience the pedestrian attribute recognition pipeline online](https://aistudio.baidu.com/community/app/387978/webUI?source=appCenter) by recognizing the demo images provided by the official platform, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/pedestrian_attribute_recognition/ped_attr_aistudio.png"/>

If you are satisfied with the performance of the pipeline, you can directly integrate and deploy it. You can choose to download the deployment package from the cloud, or refer to the methods in [Section 2.2 Local Experience](#22-local-experience) for local deployment. If you are not satisfied with the effect, you can <b>fine-tune the models in the pipeline using your private data</b>. If you have local hardware resources for training, you can start training directly on your local machine; if not, the Star River Zero-Code platform provides a one-click training service. You don't need to write any code—just upload your data and start the training task with one click.

### 2.2 Local Experience
Before using the pedestrian attribute recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
You can quickly experience the pedestrian attribute recognition pipeline with a single command. Use [the test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg) and replace `--input` with your local path for prediction.

```bash
paddlex --pipeline pedestrian_attribute_recognition --input pedestrian_attribute_002.jpg --device gpu:0 --save_path ./output/
```

The relevant parameter descriptions can be found in the parameter explanation section of [2.2.2 Python Script Integration](#222-python脚本方式集成).

After running, the result will be printed to the terminal, as shown below:

```bash
{'res': {'input_path': 'pedestrian_attribute_002.jpg', 'boxes': [{'labels': ['Trousers(长裤)', 'Age18-60(年龄在18-60岁之间)', 'LongCoat(长外套)', 'Side(侧面)'], 'cls_scores': array([0.99965, 0.99963, 0.98866, 0.9624 ]), 'det_score': 0.9795178771018982, 'coordinate': [87.24581, 322.5872, 546.2697, 1039.9852]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Front(面朝前)', 'Age18-60(年龄在18-60岁之间)'], 'cls_scores': array([0.99996, 0.99872, 0.93379, 0.71614]), 'det_score': 0.967143177986145, 'coordinate': [737.91626, 306.287, 1150.5961, 1034.2979]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Age18-60(年龄在18-60岁之间)', 'Side(侧面)'], 'cls_scores': array([0.99996, 0.99514, 0.98726, 0.96224]), 'det_score': 0.9645745754241943, 'coordinate': [399.45944, 281.9107, 869.5312, 1038.9962]}]}}
```

For the explanation of the running result parameters, you can refer to the result interpretation in [Section 2.2.2 Integration via Python Script](#222-integration-via-python-script).

The visualization results are saved under `save_path`, and the visualization result is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/01.jpg"/>


#### 2.2.2 Integration via Python Script
* The above command line is for quick experience and viewing of results. Generally, in projects, integration through code is often required. You can complete the pipeline's fast inference with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="pedestrian_attribute_recognition")

output = pipeline.predict("pedestrian_attribute_002.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) The pedestrian attribute recognition pipeline object is instantiated via `create_pipeline()`. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of a pipeline, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying the specific card number of GPUs, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPUs, such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. This is only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) The `predict()` method of the pedestrian attribute recognition pipeline object is called to perform inference prediction. This method returns a `generator`. Below are the parameters and their descriptions for the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>The data to be predicted. It supports multiple input types and is required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code>.</li>
<li><b>str</b>: Local path of the image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of the image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code>.</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference.</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: Use CPU for inference, such as <code>cpu</code>.</li>
<li><b>GPU</b>: Use the specified GPU for inference, such as <code>gpu:0</code> for the first GPU.</li>
<li><b>NPU</b>: Use the specified NPU for inference, such as <code>npu:0</code> for the first NPU.</li>
<li><b>XPU</b>: Use the specified XPU for inference, such as <code>xpu:0</code> for the first XPU.</li>
<li><b>MLU</b>: Use the specified MLU for inference, such as <code>mlu:0</code> for the first MLU.</li>
<li><b>DCU</b>: Use the specified DCU for inference, such as <code>dcu:0</code> for the first DCU.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used. During initialization, it will prioritize the local GPU device 0; if unavailable, it will use the CPU.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_threshold</code></td>
<td>Threshold for pedestrian detection visualization.</td>
<td><code>float | None</code></td>
<td>
<ul>
<li><b>float</b>: For example, <code>0.5</code>, which means filtering out all bounding boxes with a score less than <code>0.5</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized to <code>0.5</code>.</li>
</ul>
</td>
<td><code>0.5</code></td>
</tr>
<tr>
<td><code>cls_threshold</code></td>
<td>Threshold for pedestrian attribute prediction.</td>
<td><code>float | dict | list | None</code></td>
<td>
<ul>
<li><b>float</b>: A uniform threshold for attribute recognition.</li>
<li><b>list</b>: For example, <code>[0.5, 0.45, 0.48, 0.4]</code>, which means different thresholds for different classes in the order of <code>label list</code>.</li>
<li><b>dict</b>: The key is <code>default</code> or <code>int</code> type, and the value is a <code>float</code> threshold. For example, <code>{"default": 0.5, 0: 0.45, 2: 0.48, 7: 0.4}</code>. <code>default</code> represents the uniform threshold for multi-label classification, while other <code>int</code> types apply specific thresholds to classes with cls_id 0, 2, and 7.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized to <code>0.7</code>.</li>
</ul>
</td>
<td><code>0.7</code></td>
</tr>
</table>

3) Process the prediction results. Each sample's prediction result is of type `dict`, and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file will have the same name as the input file</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supporting both directory and file paths</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal, and the content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.
    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.
    - `boxes`: `(List[Dict])` Indicates the category ID of the prediction result.
    - `labels`: `(List[str])` Indicates the category name of the prediction result.
    - `cls_scores`: `(List[numpy.ndarray])` Indicates the confidence of the attribute prediction result.
    - `det_scores`: `(float)` Indicates the confidence of the pedestrian detection box.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` type will be converted to a list format.
- Calling the `save_to_img()` method will save the visualization result to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, and only the last image will be retained.)

* Additionally, it also supports obtaining visualized images with results and prediction results through attributes, as follows:

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
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained through the `json` attribute is of type `dict`, and its content is consistent with the result saved by the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary. The key `res` corresponds to the value of an `Image.Image` object: a visualized image displaying the attribute recognition result.

Additionally, you can obtain the pedestrian attribute recognition pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```bash
paddlex --get_pipeline_config pedestrian_attribute_recognition --save_path ./my_path
```


If you have obtained the configuration file, you can customize the settings for the pedestrian attribute recognition pipeline by simply modifying the value of the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file. The example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/pedestrian_attribute_recognition.yaml")

output = pipeline.predict(
    input="./pedestrian_attribute_002.jpg",
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are the initialization parameters for the pipeline. If you wish to change the initialization parameters for the pedestrian attribute recognition pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path to the configuration file with `--pipeline`.

## 3. Development Integration/Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline directly into your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In practical production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure the efficient operation of the system and a smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed information on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common form of deployment in practical production environments. By encapsulating the inference functionality into a service, clients can access these services via network requests to obtain inference results. PaddleX supports multiple service-oriented deployment solutions for pipelines. For detailed information on service-oriented deployment, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-oriented deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Get pedestrian attribute recognition results.</p>
<p><code>POST /pedestrian-attribute-recognition</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of an image file accessible by the server or the Base64-encoded content of an image file.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>detThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>det_threshold</code> parameter description in the pipeline <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>clsThreshold</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>Refer to the <code>cls_threshold</code> parameter description in the pipeline <code>predict</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pedestrians</code></td>
<td><code>array</code></td>
<td>Information about the location and attributes of pedestrians.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code>| <code>null</code></td>
<td>The result image of pedestrian attribute recognition. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>pedestrians</code> is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>The location of the pedestrian. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box.</td>
</tr>
<tr>
<td><code>attributes</code></td>
<td><code>array</code></td>
<td>The attributes of the pedestrian.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The detection score.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>attributes</code> is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>The attribute label.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The classification score.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-Language Service Call Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/pedestrian-attribute-recognition" # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode the local image using Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the returned data
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected pedestrians:")
print(result["pedestrians"])
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md). You can choose the appropriate deployment method based on your needs to integrate the model pipeline into your AI applications.

## 4. Custom Development
If the default model weights provided by the pedestrian attribute recognition pipeline are not satisfactory in terms of accuracy or speed for your specific scenario, you can attempt to further <b>fine-tune</b> the existing models using <b>your own domain-specific or application data</b> to improve the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the pedestrian attribute recognition pipeline includes both a pedestrian attribute recognition module and a pedestrian detection module, if the pipeline's performance does not meet expectations, the issue may stem from either module. You can analyze the images with poor recognition results to determine which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-Tuning Module</th>
<th>Fine-Tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate pedestrian detection</td>
<td>Pedestrian Detection Module</td>
<td><a href="../../../module_usage/tutorials/cv_modules/human_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate attribute recognition</td>
<td>Pedestrian Attribute Recognition Module</td>
<td><a href="../../../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.en.md">Link</a></td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After you complete fine-tuning with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding position in the file:

```yaml
pipeline_name: pedestrian_attribute_recognition

SubModules:
  Detection:
    module_name: object_detection
    model_name: PP-YOLOE-L_human
    model_dir: null # Replace with the path to the fine-tuned pedestrian detection model weights
    batch_size: 1
    threshold: 0.5
  Classification:
    module_name: multilabel_classification
    model_name: PP-LCNet_x1_0_pedestrian_attribute
    model_dir: null # Replace with the path to the fine-tuned pedestrian attribute recognition model weights
    batch_size: 1
    threshold: 0.7
```

Subsequently, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the --device parameter</b> to seamlessly switch between different hardware devices.
For example, if you are using Ascend NPU for inference in the pedestrian attribute recognition pipeline, the Python command you would use is:

```bash
paddlex --pipeline pedestrian_attribute_recognition \
        --input pedestrian_attribute_002.jpg \
        --device npu:0
```

If you want to use the general Pedestrian Attribute Recognition pipeline on a wider range of hardware devices, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
