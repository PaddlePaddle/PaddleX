---
comments: true
---

# Pedestrian Attribute Recognition Pipeline Tutorial

## 1. Introduction to Pedestrian Attribute Recognition Pipeline
Pedestrian attribute recognition is a key function in computer vision systems, used to locate and label specific characteristics of pedestrians in images or videos, such as gender, age, clothing color, and style. This task not only requires accurately detecting pedestrians but also identifying detailed attribute information for each pedestrian. The pedestrian attribute recognition pipeline is an end-to-end serial system for locating and recognizing pedestrian attributes, widely used in smart cities, security surveillance, and other fields, significantly enhancing the system's intelligence level and management efficiency.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/01.jpg">

<b>The pedestrian attribute recognition pipeline includes a pedestrian detection module and a pedestrian attribute recognition module</b>, with several models in each module. Which models to use specifically can be selected based on the benchmark data below. <b>If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model storage size, choose models with smaller storage</b>.


<p><b>Pedestrian Detection Module</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE-L_human</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_human_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_human_pretrained.pdparams">Trained Model</a></td>
<td>48.0</td>
<td>81.9</td>
<td>32.8</td>
<td>777.7</td>
<td>196.02</td>
<td rowspan="2">Pedestrian detection model based on PP-YOLOE</td>
</tr>
<tr>
<td>PP-YOLOE-S_human</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_human_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_human_pretrained.pdparams">Trained Model</a></td>
<td>42.5</td>
<td>77.9</td>
<td>15.0</td>
<td>179.3</td>
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
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_pedestrian_attribute</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams">Trained Model</a></td>
<td>92.2</td>
<td>3.84845</td>
<td>9.23735</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_pedestrian_attribute is a lightweight pedestrian attribute recognition model based on PP-LCNet, covering 26 categories.</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are mA on PaddleX's internally built dataset. GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX can quickly demonstrate their effectiveness. You can experience the pedestrian attribute recognition pipeline online or locally using command line or Python.

### 2.1 Online Experience
Not supported yet.

### 2.2 Local Experience
Before using the pedestrian attribute recognition pipeline locally, ensure you have completed the installation of the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
You can quickly experience the pedestrian attribute recognition pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg) and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline pedestrian_attribute_recognition --input pedestrian_attribute_002.jpg --device gpu:0
```
Parameter Description:

```
--pipeline: The name of the pipeline, here it is the pedestrian attribute recognition pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 means using the first GPU, gpu:1,2 means using the second and third GPUs), or you can choose to use CPU (--device cpu).
```

When executing the above Python script, the default pedestrian attribute recognition pipeline configuration file is loaded. If you need a custom configuration file, you can run the following command to obtain it:

<details><summary> 👉Click to Expand</summary>

<pre><code>paddlex --get_pipeline_config pedestrian_attribute_recognition
</code></pre>
<p>After execution, the pedestrian attribute recognition pipeline configuration file will be saved in the current path. If you wish to specify a custom save location, you can run the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code>paddlex --get_pipeline_config pedestrian_attribute_recognition --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, you can replace <code>--pipeline</code> with the saved path of the configuration file to make it effective. For example, if the configuration file is saved at <code>./pedestrian_attribute_recognition.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./pedestrian_attribute_recognition.yaml --input pedestrian_attribute_002.jpg --device gpu:0
</code></pre>
<p>Among them, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified, and the parameters in the configuration file will be used. If parameters are still specified, the specified parameters will take precedence.</p></details>

#### 2.2.2 Python Script Integration
A few lines of code are sufficient for quick inference of the pipeline. Taking the pedestrian attribute recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="pedestrian_attribute_recognition")

output = pipeline.predict("pedestrian_attribute_002.jpg")
for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_img("./output/")  ## Save the visualized image of the result
    res.save_to_json("./output/")  ## Save the structured output of the prediction
```
The results obtained are the same as those from the command line approach.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td>"gpu"</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `predict` method of the pedestrian attribute recognition pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods. Specific examples are as follows:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing in Python variables, such as image data represented by numpy.ndarray.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the file path of the data to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in a local directory, which should contain the data files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing in a dictionary type, where the key needs to correspond to the specific task, such as "img" for the pedestrian attribute recognition task, and the value of the dictionary supports the above data types, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing in a list, where the elements of the list need to be the above data types, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

(4) Process the prediction results: The prediction result for each sample is of type `dict` and supports printing or saving as a file. The supported save types are related to the specific pipeline, such as:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Method Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>print</code></td>
<td>Print the results to the terminal</td>
<td><code>- format_json</code>: bool, whether to format the output content with json indentation, default is True;<br/><code>- indent</code>: int, json formatting setting, only effective when <code>format_json</code> is True, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, only effective when <code>format_json</code> is True, default is False;</td>
</tr>
<tr>
<td><code>save_to_json</code></td>
<td>Save the results as a json-formatted file</td>
<td><code>- save_path</code>: str, the path to save the file, when it is a directory, the saved file name is consistent with the input file name;<br/><code>- indent</code>: int, json formatting setting, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, default is False;</td>
</tr>
<tr>
<td><code>save_to_img</code></td>
<td>Save the results as an image file</td>
<td></td>
</tr>
</tbody>
</table>
If you have obtained the configuration file, you can customize various configurations for the pedestrian attribute recognition pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of your pipeline configuration file.

For example, if your configuration file is saved as `./my_path/pedestrian_attribute_recognition*.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/pedestrian_attribute_recognition.yaml")
output = pipeline.predict("pedestrian_attribute_002.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the visualized result image
    res.save_to_json("./output/")  # Save the structured output of the prediction
```
## 3. Development Integration/Deployment
If the pedestrian attribute recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to directly apply the pedestrian attribute recognition pipeline in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

<details><summary>API Reference</summary>

<p>For primary operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and the response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body properties are as follows:</li>
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
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the response body properties are as follows:</li>
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
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>Primary operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Get pedestrian attribute recognition results.</p>
<p><code>POST /pedestrian-attribute-recognition</code></p>
<ul>
<li>The request body properties are as follows:</li>
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
<td>The URL of an image file accessible by the server or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following properties:</li>
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
<td>Information about the pedestrian's location and attributes.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The pedestrian attribute recognition result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>pedestrians</code> is an <code>object</code> with the following properties:</p>
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
<td>The location of the pedestrian. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively.</td>
</tr>
<tr>
<td><code>attributes</code></td>
<td><code>array</code></td>
<td>The pedestrian attributes.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The detection score.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>attributes</code> is an <code>object</code> with the following properties:</p>
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
<td>The label of the attribute.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The classification score.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/pedestrian-attribute-recognition&quot;
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nDetected pedestrians:&quot;)
print(result[&quot;pedestrians&quot;])
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method where computing and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose an appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.


### 4.1 Model Fine-tuning
Since the pedestrian attribute recognition pipeline includes both a pedestrian attribute recognition module and a pedestrian detection module, the unexpected performance of the pipeline may stem from either module.
You can analyze images with poor recognition results. If during the analysis, you find that many main targets are not detected, it may indicate deficiencies in the pedestrian detection model. In this case, you need to refer to the [Secondary Development](../../../module_usage/tutorials/cv_modules/human_detection.en.md#secondary-development) section in the [Human Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/human_detection.en.md) and use your private dataset to fine-tune the pedestrian detection model. If the detected main attributes are incorrectly recognized, you need to refer to the [Secondary Development](../../../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.en.md#secondary-development) section in the [Pedestrian Attribute Recognition Module Development Tutorial](../../../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.en.md) and use your private dataset to fine-tune the pedestrian attribute recognition model.

### 4.2 Model Application
After fine-tuning training with your private dataset, you will obtain local model weight files.

If you need to use the fine-tuned model weights, you only need to modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```
......
Pipeline:
  det_model: PP-YOLOE-L_human
  cls_model: PP-LCNet_x1_0_pedestrian_attribute  # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Subsequently, refer to the command-line method or Python script method in the local experience, and load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports multiple mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modifying the `--device` parameter</b>  allows seamless switching between different hardware.

For example, if you use an NVIDIA GPU for inference in the pedestrian attribute recognition pipeline, the command used is:

```bash
paddlex --pipeline pedestrian_attribute_recognition --input pedestrian_attribute_002.jpg --device gpu:0
```
At this point, if you want to switch the hardware to Ascend NPU, you only need to change `--device` to npu:0:

```bash
paddlex --pipeline pedestrian_attribute_recognition --input pedestrian_attribute_002.jpg --device npu:0
```
If you want to use the pedestrian attribute recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
