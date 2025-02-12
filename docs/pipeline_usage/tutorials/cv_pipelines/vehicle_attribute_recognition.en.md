---
comments: true
---

# Vehicle Attribute Recognition Pipeline Tutorial

## 1. Introduction to Vehicle Attribute Recognition Pipeline
Vehicle attribute recognition is a crucial component in computer vision systems. Its primary task is to locate and label specific attributes of vehicles in images or videos, such as vehicle type, color, and license plate number. This task not only requires accurately detecting vehicles but also identifying detailed attribute information for each vehicle. The vehicle attribute recognition pipeline is an end-to-end serial system for locating and recognizing vehicle attributes, widely used in traffic management, intelligent parking, security surveillance, autonomous driving, and other fields. It significantly enhances system efficiency and intelligence levels, driving the development and innovation of related industries. This pipeline also offers a flexible service-oriented deployment approach, supporting the use of multiple programming languages on various hardware platforms. Moreover, this production line provides the capability for secondary development. You can train and optimize models on your own dataset based on this production line, and the trained models can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/vehicle_attribute_recognition/01.jpg"/>
<b>The vehicle attribute recognition pipeline includes a vehicle detection module and a vehicle attribute recognition module</b>, with several models in each module. Which models to use can be selected based on the benchmark data below. <b>If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model storage size, choose models with smaller storage</b>.

<p><b>Vehicle Detection Module</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP 0.5:0.95</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE-S_vehicle</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_vehicle_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_vehicle_pretrained.pdparams">Trained Model</a></td>
<td>61.3</td>
<td>9.79 / 3.48</td>
<td>54.14 / 46.69</td>
<td>28.79</td>
<td rowspan="2">Vehicle detection model based on PP-YOLOE</td>
</tr>
<tr>
<td>PP-YOLOE-L_vehicle</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_vehicle_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_vehicle_pretrained.pdparams">Trained Model</a></td>
<td>63.9</td>
<td>32.84 / 9.03</td>
<td>176.60 / 176.60</td>
<td>196.02</td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are mAP(0.5:0.95) on the PPVehicle validation set. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Vehicle Attribute Recognition Module</b>:</p>
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
<td>PP-LCNet_x1_0_vehicle_attribute</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_vehicle_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_vehicle_attribute_pretrained.pdparams">Trained Model</a></td>
<td>91.7</td>
<td>2.32 / 0.52</td>
<td>3.22 / 1.26</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_vehicle_attribute is a lightweight vehicle attribute recognition model based on PP-LCNet.</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are mA on the VeRi dataset. GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
The pre-trained models provided by PaddleX can quickly demonstrate results. You can experience the effects of the vehicle attribute recognition pipeline online or locally using command line or Python.

### 2.1 Online Experience
Not supported yet.

### 2.2 Local Experience
Before using the vehicle attribute recognition pipeline locally, ensure you have installed the PaddleX wheel package according to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

#### 2.2.1 Experience via Command Line
You can quickly experience the vehicle attribute recognition pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg) and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline vehicle_attribute_recognition --input vehicle_attribute_002.jpg --device gpu:0 --save_path ./output/
```
Parameter Description:

```bash
{'res': {'input_path': 'vehicle_attribute_002.jpg', 'boxes': [{'labels': ['red(红色)', 'sedan(轿车)'], 'cls_scores': array([0.96375, 0.94025]), 'det_score': 0.9774094820022583, 'coordinate': [196.32553, 302.3847, 639.3131, 655.57904]}, {'labels': ['suv(SUV)', 'brown(棕色)'], 'cls_scores': array([0.99968, 0.99317]), 'det_score': 0.9705657958984375, 'coordinate': [769.4419, 278.8417, 1401.0217, 641.3569]}]}}
```

For the explanation of the running result parameters, you can refer to the result interpretation in [Section 2.2.2 Integration via Python Script](#222-integration-via-python-script).

The visualization results are saved under `save_path`, and the visualization result is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/vehicle_attribute_recognition/01.jpg"/>


#### 2.2.2 Integration via Python Script
* The above command line is for quick experience and viewing of results. Generally, in projects, integration through code is often required. You can complete the pipeline's fast inference with just a few lines of code. The inference code is as follows:

The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) The vehicle attribute recognition production line object is instantiated via `create_pipeline()`. The specific parameter descriptions are as follows:

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
<td>The name of the production line or the path to the production line configuration file. If it is the name of a production line, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the production line (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the production line name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for production line inference. It supports specifying the specific card number of GPUs, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPUs, such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. This is only available if the production line supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) The `predict()` method of the vehicle attribute recognition production line object is called to perform inference prediction. This method returns a `generator`. Below are the parameters and their descriptions for the `predict()` method:

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
<li><b>str</b>: Local path of the image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of the image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code>.</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for production line inference.</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: Use CPU for inference, such as <code>cpu</code>.</li>
<li><b>GPU</b>: Use the specified GPU for inference, such as <code>gpu:0</code> for the first GPU.</li>
<li><b>NPU</b>: Use the specified NPU for inference, such as <code>npu:0</code> for the first NPU.</li>
<li><b>XPU</b>: Use the specified XPU for inference, such as <code>xpu:0</code> for the first XPU.</li>
<li><b>MLU</b>: Use the specified MLU for inference, such as <code>mlu:0</code> for the first MLU.</li>
<li><b>DCU</b>: Use the specified DCU for inference, such as <code>dcu:0</code> for the first DCU.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the production line initialization will be used. During initialization, it will prioritize the local GPU device 0; if unavailable, it will use the CPU.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>det_threshold</code></td>
<td>Threshold for vehicle detection visualization.</td>
<td><code>float | None</code></td>
<td>
<ul>
<li><b>float</b>: For example, <code>0.5</code>, which means filtering out all bounding boxes with a score less than <code>0.5</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the production line initialization will be used, initialized to <code>0.5</code>.</li>
</ul>
</td>
<td><code>0.5</code></td>
</tr>
<tr>
<td><code>cls_threshold</code></td>
<td>Threshold for vehicle attribute prediction.</td>
<td><code>float | dict | list | None</code></td>
<td>
<ul>
<li><b>float</b>: A uniform threshold for attribute recognition.</li>
<li><b>list</b>: For example, <code>[0.5, 0.45, 0.48, 0.4]</code>, which means different thresholds for different classes in the order of <code>label list</code>.</li>
<li><b>dict</b>: The key is <code>default</code> or <code>int</code> type, and the value is a <code>float</code> threshold. For example, <code>{"default": 0.5, 0: 0.45, 2: 0.48, 7: 0.4}</code>. <code>default</code> represents the uniform threshold for attribute recognition, while other <code>int</code> types apply specific thresholds to classes with cls_id 0, 2, and 7.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the production line initialization will be used, initialized to <code>0.7</code>.</li>
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
<td rowspan="3">Print the result to the terminal.</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation.</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. This is only effective when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file.</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the result. When specified as a directory, the saved file will have the same name as the input file.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. This is only effective when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file.</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the result, supporting both directory and file paths.</td>
<td>None</td>
</tr>
</table>

- When calling the <code>print()</code> method, the result will be printed to the terminal. The printed content is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.
    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.
    - `boxes`: `(List[Dict])` The category IDs of the prediction results.
    - `labels`: `(List[str])` The category names of the prediction results.
    - `cls_scores`: `(List[numpy.ndarray])` The confidence scores of the attribute prediction results.
    - `det_scores`: `(float)` The confidence scores of the vehicle detection boxes.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` type will be converted to a list format.
- Calling the `save_to_img()` method will save the visualization result to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The production line usually contains many result images, so it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, and only the last image will be retained.)

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

Additionally, you can obtain the vehicle attribute recognition pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```bash
paddlex --get_pipeline_config vehicle_attribute_recognition --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the Vehicle Attribute Recognition pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the configuration file. The example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/vehicle_attribute_recognition.yaml")

output = pipeline.predict(
    input="./vehicle_attribute_002.jpg",
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are initialization parameters for the production line. If you wish to change the initialization parameters for the vehicle attribute recognition production line, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file by specifying the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment

If the production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the production line into your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python-script-method).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In practical production environments, many applications have strict standards for the performance metrics of deployment strategies, especially response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX offers a high-performance inference plugin aimed at deeply optimizing the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common form of deployment in practical production environments. By encapsulating inference capabilities into services, clients can access these services via network requests to obtain inference results. PaddleX supports various production line service-oriented deployment solutions. For detailed production line service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-oriented deployment and multi-language service invocation examples:

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
<p>Get the vehicle attribute recognition results.</p>
<p><code>POST /vehicle-attribute-recognition</code></p>
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
<td>The URL of the image file accessible by the server or the Base64-encoded content of the image file.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following attributes:</li>
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
<td><code>vehicles</code></td>
<td><code>array</code></td>
<td>Information about the vehicle's location and attributes.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The vehicle attribute recognition result image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>vehicles</code> is an <code>object</code> with the following attributes:</p>
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
<td>The location of the vehicle. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.</td>
</tr>
<tr>
<td><code>attributes</code></td>
<td><code>array</code></td>
<td>The attributes of the vehicle.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Detection score.</td>
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
<td>Classification score.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-Language Service Call Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/vehicle-attribute-recognition" # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode the local image using Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected vehicles:")
print(result["vehicles"])
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data locally without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model production line into your AI application.

## 4. Secondary Development
If the default model weights provided by the vehicle attribute recognition production line do not meet your requirements in terms of accuracy or speed, you can attempt to <b>further fine-tune the existing models using your own domain-specific or application-specific data</b> to improve the recognition performance of the vehicle attribute recognition production line in your scenario.

### 4.1 Model Fine-Tuning
Since the vehicle attribute recognition production line includes both a vehicle detection module and a vehicle attribute recognition module, if the performance of the production line is not satisfactory, the issue may lie in either of these modules. You can analyze the images with poor recognition results to determine which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

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
<td>Inaccurate vehicle detection</td>
<td>Vehicle Detection Module</td>
<td><a href="../../../module_usage/tutorials/cv_modules/vehicle_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate attribute recognition</td>
<td>Vehicle Attribute Recognition Module</td>
<td><a href="../../../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.en.md">Link</a></td>
</tr>
</tbody>
</table>
<b>Note:</b> The parameters in the configuration file are initialization parameters for the production line. If you wish to change the initialization parameters for the vehicle attribute recognition production line, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file by specifying the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment

If the production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the production line into your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python-script-method).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In practical production environments, many applications have strict standards for the performance metrics of deployment strategies, especially response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX offers a high-performance inference plugin aimed at deeply optimizing the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common form of deployment in practical production environments. By encapsulating inference capabilities into services, clients can access these services via network requests to obtain inference results. PaddleX supports various production line service-oriented deployment solutions. For detailed production line service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-oriented deployment and multi-language service invocation examples:

```yaml
pipeline_name: vehicle_attribute_recognition

SubModules:
  Detection:
    module_name: object_detection
    model_name: PP-YOLOE-L_vehicle
    model_dir: null # Replace with the path to the fine-tuned vehicle detection model weights
    batch_size: 1
    threshold: 0.5
  Classification:
    module_name: multilabel_classification
    model_name: PP-LCNet_x1_0_vehicle_attribute
    model_dir: null # Replace with the path to the fine-tuned vehicle attribute recognition model weights
    batch_size: 1
    threshold: 0.7
```

Subsequently, you can load the modified pipeline configuration file using the command line or Python script methods described in the local experience section.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you are using Ascend NPU for vehicle attribute recognition inference, the Python command is as follows:

```bash
paddlex --pipeline vehicle_attribute_recognition \
        --input vehicle_attribute_002.jpg \
        --device npu:0
```

If you want to use the general Vehicle Attribute Recognition pipeline on a wider range of hardware devices, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
