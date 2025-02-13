---
comments: true
---

# Tutorial on Using the Rotated Object Detection Pipeline

## 1. Introduction to the Rotated Object Detection Pipeline
Rotated object detection is a variant of the object detection module, specifically designed for detecting rotated objects. Rotated bounding boxes are often used to detect rectangular boxes with angular information, where the width and height of the box are no longer parallel to the image coordinate axes. Compared to horizontal rectangular boxes, rotated rectangular boxes generally include less background information. Rotated object detection has important applications in remote sensing scenarios. This pipeline also provides flexible service deployment options, supporting multiple programming languages on various hardware. Moreover, this pipeline offers secondary development capabilities, allowing you to train and fine-tune models on your own dataset, with seamless integration of the trained models.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/rotated_object_detection/rotated_object_detection_001_res.png">

<b>The rotated object detection pipeline includes a rotated object detection module, which contains multiple models. You can choose the model based on the benchmark data provided below.</b>

<b>If you prioritize model accuracy, choose a model with higher accuracy; if you care more about inference speed, choose a model with faster inference speed; if you are concerned about model storage size, choose a model with a smaller storage size.</b>

<p><b>Image Rotated Object Detection Module (Optional):</b></p>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE-R-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-R-L_infer.tar">Inference Model</a>/<a href="https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams">Training Model</a></td>
<td>78.14</td>
<td>20.7039</td>
<td>157.942</td>
<td>211.0 M</td>
<td rowspan="1">PP-YOLOE-R is an efficient one-stage anchor-free rotated bounding box detection model. Based on PP-YOLOE, PP-YOLOE-R introduces several useful designs to improve detection accuracy with minimal additional parameters and computational cost.</td>
</tr>
</table>

<p><b>Note: The above accuracy metrics are based on the <a href="https://captain-whu.github.io/DOTA/">DOTA</a> validation set mAP(0.5:0.95). All model GPU inference times are based on NVIDIA TRX2080 Ti machines with F16 precision, and CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start

### 2.1 Local Experience
> ❗ Before using the rotated object detection pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.1.1 Command Line Experience
* You can quickly experience the rotated object detection pipeline with a single command. Use the [test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png) and replace `--input` with your local path for prediction.

Due to network issues, the above web page parsing was not successful. If you need the content of the web page, please check the validity of the web page link and try again. If you do not need the parsing of this link, you can proceed with other questions.

```bash
paddlex --pipeline rotated_object_detection \
        --input rotated_object_detection_001.png \
        --threshold 0.5 \
        --save_path ./output \
        --device gpu:0 \
```

The relevant parameter descriptions can be referred to in the parameter explanations of [2.2.2 Integration via Python Script](#222-integration-via-python-script).

After running, the results will be printed to the terminal, as follows:

```bash
{'res': {'input_path': 'rotated_object_detection_001.png', 'page_index': None, 'boxes': [{'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7409099340438843, 'coordinate': [92.88687, 763.1569, 85.163124, 749.5868, 116.07975, 731.99414, 123.8035, 745.5643]}, {'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7393015623092651, 'coordinate': [348.2332, 177.55974, 332.77704, 150.24973, 345.2183, 143.21028, 360.67444, 170.5203]}, {'cls_id': 11, 'label': 'roundabout', 'score': 0.8101699948310852, 'coordinate': [537.1732, 695.5475, 204.4297, 612.9735, 286.71338, 281.48022, 619.4569, 364.05426]}]}}
```

The explanation of the result parameters can be found in [2.2.2 Python Script Integration](#222-python-script-integration).

The visualized results are saved under `save_path`, and the visualized result of rotated object detection is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/rotated_object_detection/rotated_object_detection_001_res.png">

#### 2.1.2 Python Script Integration
* The above command line is for quickly experiencing and viewing the effect. Generally, in a project, you often need to integrate through code. You can complete the quick inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline_name="rotated_object_detection")
output = pipeline.predict(input="rotated_object_detection_001.png", threshold=0.5)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps were executed:

(1) The Rotated Object Detection pipeline object was instantiated via `create_pipeline()`, with the specific parameters described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline_name</code></td>
<td>The name of the pipeline, which must be supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>The path to the pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying the specific card number of the GPU, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu".</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, which is only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) The `predict()` method of the Rotated Object Detection pipeline object was called for inference prediction. This method returns a `generator`. Below are the parameters of the `predict()` method and their descriptions:

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
<td>Data to be predicted, supporting multiple input types (required).</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
  <li><b>str</b>: Local path of image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png">Example</a>; <b>Local directory</b>, the directory should contain images to be predicted, e.g., local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with a specific file path)</li>
  <li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: e.g., <code>cpu</code> indicates using CPU for inference;</li>
  <li><b>GPU</b>: e.g., <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
  <li><b>NPU</b>: e.g., <code>npu:0</code> indicates using the 1st NPU for inference;</li>
  <li><b>XPU</b>: e.g., <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
  <li><b>MLU</b>: e.g., <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
  <li><b>DCU</b>: e.g., <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used. During initialization, the local GPU 0 will be prioritized; if unavailable, the CPU will be used;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>The actual image resolution used during model inference</td>
<td><code>None|float|dict[int, float]</code></td>
<td>
<ul>
    <li><b>None</b>: If set to <code>None</code>, the default pipeline initialization parameter <code>0.5</code> will be used, i.e., 0.5 as the low-score object filtering threshold for all categories</li>
    <li><b>float</b>: Any float number greater than 0 and less than 1</li>
    <li><b>dict[int, float]</b>: The key represents the category ID, and the value represents the threshold for that category, allowing different low-score filtering thresholds for different categories, e.g., <code>{0:0.5, 1:0.35}</code> indicates using 0.5 and 0.35 as the low-score filtering thresholds for categories 0 and 1, respectively</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

(3) Process the prediction results. The prediction result for each sample is of the `dict` type and supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal, with the printed content explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates which page of the PDF it is, otherwise it is `None`

    - `boxes`: `(list)` Detection box information, each element is a dictionary containing the following fields
      - `cls_id`: `(int)` Category ID
      - `label`: `(str)` Category name
      - `score`: `(float)` Confidence score
      - `coordinates`: `(list)` Detection box coordinates, in the format `[x1, y1, x2, y2, x3, y3, x4, y4]`

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.json`; if specified as a file, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` types will be converted to lists.

- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`; if specified as a file, it will be saved directly to that file.

* Additionally, it also supports obtaining visualized images and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the predicted <code>json</code> format result</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is a dict type of data, with content consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type of data. The key is `res`, and the corresponding value is an `Image.Image` object: an image used to display the prediction result of rotated object detection.

In addition, you can obtain the rotated object detection pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config rotated_object_detection --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the Rotated Object Detection Pipeline. Simply modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. The example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/rotated_object_detection.yaml")

output = pipeline.predict(
    input="./rotated_object_detection_001.png",
    threshold=0.5,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are the initialization parameters for the pipeline. If you want to change the initialization parameters of the Rotated Object Detection Pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in the configuration file by specifying the path with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.2 Integration via Python Script](#22-integration-via-python-script).

In addition, PaddleX also provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent performance requirements (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. Therefore, PaddleX provides a high-performance inference plugin designed to deeply optimize the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services via network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references and multi-language service call examples for basic service deployment:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body has the following properties:</li>
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
<td>Error code. Fixed to <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed to <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the response body has the following properties:</li>
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
<p>Perform object detection on the image.</p>
<p><code>POST /rotated-object-detection</code></p>
<ul>
<li>The request body has the following properties:</li>
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
<td>The URL of an image file accessible to the server or the Base64 encoded result of the image file content.</td>
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
<li>When the request is processed successfully, the <code>result</code> property of the response body has the following properties:</li>
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
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>Information about the position, category, etc., of the objects.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>Object detection result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>detectedObjects</code> is an <code>object</code> with the following properties:</p>
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
<td>Object position. The elements in the array are the x-coordinate of the top-left corner, y-coordinate of the top-left corner, x-coordinate of the bottom-right corner, and y-coordinate of the bottom-right corner of the bounding box.</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>Object category ID.</td>
</tr>
<tr>
<td><code>categoryName</code></td>
<td><code>string</code></td>
<td>The name of the target category.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Object score.</td>
</tr>
</tbody>
</table>
<p>An example of the <code>result</code> is as follows:</p>
<pre><code class="language-json">{
"detectedObjects": [
{
"bbox": [
92.88687133789062,
763.1569213867188,
85.16312408447266,
749.5867919921875,
116.07975006103516,
731.994140625,
123.80349731445312,
745.5642700195312
],
"categoryId": 0,
"score": 0.7418514490127563
},
{
"bbox": [
348.2331848144531,
177.5597381591797,
332.77703857421875,
150.24972534179688,
345.2182922363281,
143.2102813720703,
360.6744384765625,
170.52029418945312
],
"categoryId": 1,
"score": 0.7328268885612488
}
],
"image": "xxxxxx"
}
</code></pre></details>

<details><summary>Multi-language Service Invocation Example</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">
import base64
import requests

API_URL = "http://localhost:8080/rotated-object-detection"  # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode the local image with Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the returned data from the interface
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected objects:")
print(result["detectedObjects"])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // <url id="cu9pu8852ceh1d3h24gg" type="url" status="parsed" title="GitHub - Huiyicc/cpp-httplib: A C++ header-only HTTP/HTTPS server and client library" wc="15064">https://github.com/Huiyicc/cpp-httplib</url>
#include &quot;nlohmann/json.hpp&quot; // <url id="cu9pu8852ceh1d3h24h0" type="url" status="parsed" title="GitHub - nlohmann/json: JSON for Modern C++" wc="80311">https://github.com/nlohmann/json</url>
#include &quot;base64.hpp&quot; // <url id="cu9pu8852ceh1d3h24hg" type="url" status="parsed" title="GitHub - tobiaslocker/base64: A modern C++ base64 encoder / decoder" wc="2293">https://github.com/tobiaslocker/base64</url>

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // Encode the local image with Base64
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;image&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post(&quot;/small-object-detection&quot;, headers, body, &quot;application/json&quot;);
    // Process the returned data from the interface
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto detectedObjects = result[&quot;detectedObjects&quot;];
        std::cout &lt;&lt; &quot;\nDetected objects:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : detectedObjects) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>

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

        // Create an OkHttpClient instance
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

    // Encode the local image using Base64
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
    client := &http.Client{}
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
            Image          string   `json:"image"`
            DetectedObjects []map[string]interface{} `json:"detectedObjects"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &respData)
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

        // Process the response data
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

const API_URL = 'http://localhost:8080/small-object-detection';
const imagePath = './demo.jpg';
const outputImagePath = "./out.jpg";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64-encoded file content or image URL
  })
};

// Encode the local image in Base64
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) => {
    // Process the response data
    const result = response.data["result"];
    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) => {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log("\nDetected objects:");
    console.log(result["detectedObjects"]);
})
.catch((error) => {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/small-object-detection"; // Service URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// Encode the local image in Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data); // Base64-encoded file content or image URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the response data from the API
$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nDetected objects:\n";
print_r($result["detectedObjects"]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data locally without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.


## 4. Secondary Development
If the default model weights provided by the Rotated Object Detection Pipeline do not meet your requirements in terms of accuracy or speed, you can attempt to <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to improve the detection performance in your scenario.

### 4.1 Model Fine-Tuning
Since the Rotated Object Detection Pipeline includes a rotated object detection module, if the pipeline's performance is not satisfactory, you can analyze the poorly detected images and refer to the fine-tuning tutorial links in the table below for model fine-tuning.


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
      <td>Prediction results are not satisfactory</td>
      <td>Rotated Object Detection Module</td>
      <td><a href="../../../module_usage/tutorials/cv_modules/rotated_object_detection.en.md">Link</a></td>
    </tr>
  </tbody>
</table>

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain the local model weight file.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the path of the fine-tuned model weights with the corresponding location in the pipeline configuration file:

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/small-object-detection"; // Service URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// Encode the local image in Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data); // Base64-encoded file content or image URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the response data from the API
$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nDetected objects:\n";
print_r($result["detectedObjects"]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data locally without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.


## 4. Secondary Development
If the default model weights provided by the Rotated Object Detection Pipeline do not meet your requirements in terms of accuracy or speed, you can attempt to <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to improve the detection performance in your scenario.

### 4.1 Model Fine-Tuning
Since the Rotated Object Detection Pipeline includes a rotated object detection module, if the pipeline's performance is not satisfactory, you can analyze the poorly detected images and refer to the fine-tuning tutorial links in the table below for model fine-tuning.


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
      <td>Prediction results are not satisfactory</td>
      <td>Rotated Object Detection Module</td>
      <td><a href="../../../module_usage/tutorials/cv_modules/rotated_object_detection.en.md">Link</a></td>
    </tr>
  </tbody>
</table>

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain the local model weight file.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the path of the fine-tuned model weights with the corresponding location in the pipeline configuration file:

```yaml
SubModules:
  RotatedObjectDetection:
    module_name: rotated_object_detection
    model_name: PP-YOLOE-R-L
    model_dir: null # Here replaced with the newly fine-tuned weight path.
    batch_size: 1
    threshold: 0.5
```

Subsequently, refer to the command line method or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to achieve seamless switching between different hardware.

For example, if you use Ascend NPU for inference with the Rotated Object Detection pipeline, the Python command is:

```bash
paddlex --pipeline rotated_object_detection \
        --input rotated_object_detection_001.png \
        --threshold 0.5 \
        --save_path ./output \
        --device npu:0
```

If you want to use rotated object detection on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
