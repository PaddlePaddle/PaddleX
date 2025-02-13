---
comments: true
---

# Small Object Detection Pipeline Tutorial

## 1. Introduction to Small Object Detection Pipeline
Small object detection is a specialized technique for identifying tiny objects within images, widely applied in fields such as surveillance, autonomous driving, and satellite image analysis. It can accurately locate and classify small-sized objects like pedestrians, traffic signs, or small animals within complex scenes. By leveraging deep learning algorithms and optimized Convolutional Neural Networks (CNNs), small object detection significantly enhances the recognition capabilities for small objects, ensuring no critical information is overlooked in practical applications. This technology plays a pivotal role in enhancing safety and automation levels.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/01.png"/>
<b>The small object detection pipeline includes a small object detection module. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, opt for a model with a smaller storage size.</b>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE_plus_SOD-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">Trained Model</a></td>
<td>25.1</td>
<td>135.68 / 122.94</td>
<td>188.09 / 107.74</td>
<td>77.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">Trained Model</a></td>
<td>31.9</td>
<td>114.24 / 93.98</td>
<td>285.39 / 285.39</td>
<td>325.0 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-largesize-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">Trained Model</a></td>
<td>42.7</td>
<td>639.57 / 332.79</td>
<td>2807.12 / 2807.12</td>
<td>340.5 M</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are based on the </b><a href="https://github.com/VisDrone/VisDrone-Dataset">VisDrone-DET</a><b> validation set mAP(0.5:0.95). All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start

All model pipelines provided by PaddleX can be quickly experienced. You can experience the effect of the small object detection pipeline on the community platform, or you can use the command line or Python locally to experience the effect of the small object detection pipeline.

### 2.1 Online Experience

You can [experience the small object detection pipeline online](https://aistudio.baidu.com/community/app/387975/webUI?source=appCenter) by recognizing the demo images provided by the official platform, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/small_obj_det_aistudio.jpg"/>

If you are satisfied with the performance of the pipeline, you can directly integrate and deploy it. You can choose to download the deployment package from the cloud, or refer to the methods in [Section 2.2 Local Experience](#22-local-experience) for local deployment. If you are not satisfied with the effect, you can <b>fine-tune the models in the pipeline using your private data</b>. If you have local hardware resources for training, you can start training directly on your local machine; if not, the Star River Zero-Code platform provides a one-click training service. You don't need to write any code—just upload your data and start the training task with one click.

### 2.2 Local Experience
Before using the small object detection pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
* You can quickly experience the small object detection pipeline effect with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg), and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline small_object_detection \
        --input small_object_detection.jpg \
        --threshold 0.5 \
        --save_path ./output \
        --device gpu:0
```

The relevant parameter descriptions can be referred to in the parameter explanations in [2.2.2 Python Script Integration](#222-python-script-integration).

After running, the result will be printed to the terminal as follows:

```bash
{'res': {'input_path': 'small_object_detection.jpg', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'pedestrian', 'score': 0.8182944655418396, 'coordinate': [203.60147, 701.3809, 224.2007, 743.8429]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.8150849342346191, 'coordinate': [185.01398, 710.8665, 201.76335, 744.9308]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7748839259147644, 'coordinate': [295.1978, 500.2161, 309.33438, 532.0253]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7688254714012146, 'coordinate': [851.5233, 436.13293, 863.2146, 466.8981]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.689735472202301, 'coordinate': [802.1584, 460.10693, 815.6586, 488.85086]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6697502136230469, 'coordinate': [479.947, 309.43323, 489.1534, 332.5485]}, ...]}}
```

The explanation of the result parameters can be referred to in [2.2.2 Python Script Integration](#222-python脚本方式集成).

The visualization results are saved under `save_path`, and the visualization result of small object detection is as follows:
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/02.png"/>

#### 2.1.2 Python Script Integration
* The above command line is for quickly experiencing and viewing the effect. Generally, in a project, it is often necessary to integrate through code. You can complete the quick inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="small_object_detection")
output = pipeline.predict(input="small_object_detection.jpg", threshold=0.5)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps are performed:

(1) Instantiate the small object detection pipeline object through `create_pipeline()`, with specific parameter descriptions as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>Pipeline name or path to pipeline config file, if it's set as a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with the <code>pipeline</code>, it takes precedence over the <code>pipeline</code>, and the pipeline name must match the <code>pipeline</code>).
</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Pipeline inference device. Supports specifying the specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU such as "cpu".</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the small object detection pipeline object for inference prediction. This method will return a `generator`. The parameters of the `predict()` method and their descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of an image file or PDF file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as a network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code> (currently does not support prediction of directories containing PDF files, PDF files need to be specified to specific file paths)</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Pipeline inference device</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: Such as <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: Such as <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
<li><b>NPU</b>: Such as <code>npu:0</code> indicates using the 1st NPU for inference;</li>
<li><b>XPU</b>: Such as <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
<li><b>MLU</b>: Such as <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
<li><b>DCU</b>: Such as <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used. During initialization, it will preferentially use the local GPU 0 device, if not available, the CPU device will be used;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>threshold</code></td>
<td>Image resolution actually used during model inference</td>
<td><code>None|float|dict[int, float]</code></td>
<td>
<ul>
<li><b>None</b>: If set to <code>None</code>, the default initialization parameter <code>0.5</code> will be used, i.e., 0.5 will be used as the low-score object filtering threshold for all categories</li>
<li><b>float</b>: Any floating-point number greater than 0 and less than 1</li>
<li><b>dict[int, float]</b>: The key represents the category ID, and the value represents the threshold corresponding to the category, indicating different low-score filtering thresholds for different categories, such as <code>{0:0.5, 1:0.35}</code> means using 0.5 and 0.35 as low-score filtering thresholds for category 0 and category 1 respectively</li>
</ul>
</td>
<td><code>None</code></td>

</table>

(3) Process the prediction results. The prediction result for each sample is of type `dict` and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a json file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file name will be consistent with the input file type</td>
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
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. Supports directory or file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal. The content printed to the terminal is explained as follows:

    - `input_path`: `(str)` Input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page of the PDF; otherwise, it is `None`

    - `boxes`: `(list)` Detection box information, each element is a dictionary containing the following fields
      - `cls_id`: `(int)` Class ID
      - `label`: `(str)` Class name
      - `score`: `(float)` Confidence
      - `coordinates`: `(list)` Detection box coordinates, in the format `[xmin, ymin, xmax, ymax]`

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.json`. If specified as a file, it will be saved directly to that file. Since json files do not support saving numpy arrays, `numpy.array` types will be converted to lists.

- Calling the `save_to_img()` method will save the visualization result to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If specified as a file, it will be saved directly to that file.

* In addition, you can also obtain the visualized image and prediction results through attributes, as follows:

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

- The prediction result obtained by the `json` attribute is a dictionary type, and the content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type. The key is `res`, and the corresponding value is an `Image.Image` object: an object used to display the prediction result of small object detection.

In addition, you can obtain the small object detection pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config small_object_detection --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the small object detection pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the configuration file. An example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/small_object_detection.yaml")

output = pipeline.predict(
    input="./small_object_detection.jpg",
    threshold=0.5,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are the initialization parameters of the pipeline. If you wish to change the initialization parameters of the general small object detection pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. At the same time, CLI prediction also supports passing in the configuration file, just specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to apply the pipeline directly in your Python project, you can refer to the example code in [2.1.2 Python Script Integration](#212-python-script-integration).

In addition, PaddleX also provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin, designed to deeply optimize the performance of model inference and pre/post-processing, achieving significant acceleration of the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below is the API reference for basic service deployment and multi-language service call examples:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the properties of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
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
<li>When the request is not processed successfully, the properties of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
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
<p>Perform object detection on an image.</p>
<p><code>POST /small-object-detection</code></p>
<ul>
<li>The properties of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
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
<td><code>threshold</code></td>
<td><code>number</code> | <code>object</code> | <code>null</code></td>
<td>Refer to the <code>threshold</code> parameter description in the pipeline <code>predict</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following properties:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>Information about the location, category, and other details of detected objects.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code>| <code>null</code></td>
<td>The result image of object detection. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>detectedObjects</code> is an <code>object</code> with the following properties:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>The location of the detected object. The elements of the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>The category ID of the detected object.</td>
</tr>
<tr>
<td><code>categoryName</code></td>
<td><code>string</code></td>
<td>The name of the target category.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The score of the detected object.</td>
</tr>
</tbody>
</table>
<p><code>result</code> example is as follows:</p>
<pre><code class="language-json">{
"detectedObjects": [
{
"bbox": [
404.4967956542969,
90.15770721435547,
506.2465515136719,
285.4187316894531
],
"categoryId": 0,
"categoryName": "person",
"score": 0.7418514490127563
},
{
"bbox": [
155.33145141601562,
81.10954284667969,
199.71136474609375,
167.4235382080078
],
"categoryId": 1,
"categoryName": "bottle",
"score": 0.7328268885612488
}
],
"image": "xxxxxx"
}
</code></pre></details>
<details><summary>Multi-language Service Call Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/small-object-detection" # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode the local image using Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the API response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected objects:")
print(result["detectedObjects"])
</code></pre></details>
<details><summary>C++</summary>
<pre><code class="language-cpp">#include <iostream>
#include "cpp-httplib/httplib.h" // [GitHub - Huiyicc/cpp-httplib: A C++ header-only HTTP/HTTPS server and client library](https://github.com/Huiyicc/cpp-httplib)
#include "nlohmann/json.hpp" // [GitHub - nlohmann/json: JSON for Modern C++](https://github.com/nlohmann/json)
#include "base64.hpp" // [GitHub - tobiaslocker/base64: A modern C++ base64 encoder / decoder](https://github.com/tobiaslocker/base64)

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string outputImagePath = "./out.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // Encode the local image using Base64
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; "Error reading file." &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*="">(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post("/small-object-detection", headers, body, "application/json");
    // Process the API response data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char=""> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; "Output image saved at " &lt;&lt; outputImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; "Unable to open file for writing: " &lt;&lt; outputImagePath &lt;&lt; std::endl;
        }

        auto detectedObjects = result["detectedObjects"];
        std::cout &lt;&lt; "\nDetected objects:" &lt;&lt; std::endl;
        for (const auto&amp; category : detectedObjects) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; "Failed to send HTTP request." &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</char*></unsigned></const></char></iostream></code></pre></details>
<details><summary>Java</summary>
<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/small-object-detection"; // Service URL
        String imagePath = "./demo.jpg"; // Local image
        String outputImagePath = "./out.jpg"; // Output image

        // Encode the local image using Base64
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData); // Base64-encoded file content or image URL

        // Create OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Call the API and process the response data
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                String base64Image = result.get("image").asText();
                JsonNode detectedObjects = result.get("detectedObjects");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                System.out.println("\nDetected objects: " + detectedObjects.toString());
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
</code></pre></details>
<details><summary>Go</summary>
<pre><code class="language-go">package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/small-object-detection"
    imagePath := "./demo.jpg"
    outputImagePath := "./out.jpg"

    // Encode the local image to Base64
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64-encoded file content or image URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // Call the API
    client := &amp;http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println("Error creating request:", err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println("Error sending request:", err)
        return
    }
    defer res.Body.Close()

    // Process the response data
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Image            string   `json:"image"`
            DetectedObjects  []map[string]interface{} `json:"detectedObjects"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println("Error decoding base64 image data:", err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing image to file:", err)
        return
    }
    fmt.Printf("Image saved at %s.jpg\n", outputImagePath)
    fmt.Println("\nDetected objects:")
    for _, category := range respData.Result.DetectedObjects {
        fmt.Println(category)
    }
}
</code></pre></details>
<details><summary>C#</summary>
<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/small-object-detection";
    static readonly string imagePath = "./demo.jpg";
    static readonly string outputImagePath = "./out.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local image in Base64
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64-encoded file content or image URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the returned data
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse["result"]["image"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output image saved at {outputImagePath}");
        Console.WriteLine("\nDetected objects:");
        Console.WriteLine(jsonResponse["result"]["detectedObjects"].ToString());
    }
}
</code></pre></details>
<details><summary>Node.js</summary>
<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/small-object-detection'
const imagePath = './demo.jpg'
const outputImagePath = "./out.jpg";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64-encoded file content or image URL
  })
};

// Encode the local image using Base64
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) =&gt; {
    // Process the API response data
    const result = response.data["result"];
    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log("\nDetected objects:");
    console.log(result["detectedObjects"]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>
<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/small-object-detection"; // Service URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// Encode the local image with Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" =&gt; $image_data); // Base64 encoded file content or image URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the returned data from the interface
$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nDetected objects:\n";
print_r($result["detectedObjects"]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing functions on the user's device itself. The device can process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model into your AI application.

## 4. Secondary Development
If the default model weights provided by the general small object detection pipeline are not satisfactory in terms of accuracy or speed for your specific scenario, you can try to <b>fine-tune</b> the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the small object detection pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the general small object detection pipeline includes a small object detection module, if the pipeline's performance does not meet expectations, you can analyze the images with poor segmentation results and refer to the fine-tuning tutorial links in the table below for model fine-tuning.


<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-Tuning Module</th>
<th>Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Prediction results are not as expected</td>
<td>Small Object Detection Module</td>
<td><a href="../../../module_usage/tutorials/cv_modules/small_object_detection.en.md">Link</a></td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file and replace the local path of the fine-tuned model weights to the corresponding position in the pipeline configuration file:

```yaml
SubModules:
  SmallObjectDetection:
    module_name: small_object_detection
    model_name: PP-YOLOE_plus_SOD-L
    model_dir: null # Here replaced with the newly fine-tuned weight path.
    batch_size: 1
    threshold: 0.5
```

Subsequently, refer to the command-line method or Python script method in the [local experience](#21-local-experience) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for small object detection in the pipeline, the Python command used is:

```bash
paddlex --pipeline small_object_detection \
        --input small_object_detection.jpg \
        --threshold 0.5 \
        --save_path ./output \
        --device npu:0
```

If you want to use the universal small object detection pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
