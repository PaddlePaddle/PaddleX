---
comments: true
---

# General Image Multi-Label Classification Pipeline Tutorial

## 1. Introduction to the General Image Multi-Label Classification Pipeline
Image multi-label classification is a technique that assigns multiple relevant categories to a single image simultaneously, widely used in image annotation, content recommendation, and social media analysis. It can identify multiple objects or features present in an image, for example, an image containing both "dog" and "outdoor" labels. By leveraging deep learning models, image multi-label classification automatically extracts image features and performs accurate classification, providing users with more comprehensive information. This technology is of great significance in applications such as intelligent search engines and automatic content generation.This pipeline also offers a flexible service-oriented deployment approach, supporting the use of multiple programming languages on various hardware platforms. Moreover, this pipeline provides the capability for secondary development. You can train and optimize models on your own dataset based on this pipeline, and the trained models can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/01.png">

<b>The General Image Multi-Label Classification Pipeline includes a module for image multi-label classification. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>Model Storage Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">Trained Model</a></td>
<td>89.15</td>
<td>-</td>
</tr>
<tr>
<td>PP-HGNetV2-B0_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">Trained Model</a></td>
<td>80.98</td>
<td>39.6</td>
</tr>
<tr>
<td>PP-HGNetV2-B4_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">Trained Model</a></td>
<td>87.96</td>
<td>88.5</td>
</tr>
<tr>
<td>PP-HGNetV2-B6_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">Trained Model</a></td>
<td>91.25</td>
<td>286.5</td>
</tr>
<tr>
<td>PP-LCNet_x1_0_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_ML_pretrained.pdparams">Trained Model</a></td>
<td>77.96</td>
<td>29.4</td>
</tr>
<tr>
<td>ResNet50_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_ML_pretrained.pdparams">Trained Model</a></td>
<td>83.50</td>
<td>108.9</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are mAP for the multi-label classification task on </b><a href="https://cocodataset.org/#home">COCO2017</a><b>. The GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
All model pipelines provided by PaddleX can be quickly experienced. You can experience the effect of the image multi-label classification pipeline on the community platform, or you can use the command line or Python locally to experience the effect of the image multi-label classification pipeline.

### 2.1 Online Experience
You can [experience the image multi-label classification pipeline online](https://aistudio.baidu.com/community/app/387974/webUI?source=appCenter) by recognizing the demo images provided by the official platform, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/multi_label_cls.png"/>

If you are satisfied with the performance of the pipeline, you can directly integrate and deploy it. You can choose to download the deployment package from the cloud, or refer to the methods in [Section 2.2 Local Experience](#22-local-experience) for local deployment. If you are not satisfied with the effect, you can <b>fine-tune the models in the pipeline using your private data</b>. If you have local hardware resources for training, you can start training directly on your local machine; if not, the Star River Zero-Code platform provides a one-click training service. You don't need to write any code—just upload your data and start the training task with one click.

### 2.2 Local Experience
> ❗ Before using the image multi-label classification pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
You can quickly experience the image multi-label classification pipeline effect with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg), and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline image_multilabel_classification --input general_image_classification_001.jpg --device gpu:0
```

The relevant parameter descriptions can be referred to in the parameter explanations in [2.2.2 Python Script Integration]().

After running, the result will be printed to the terminal as follows:

```bash
{'res': {'input_path': 'test_imgs/general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([21]), 'scores': array([0.99962]), 'label_names': ['bear']}}
```

The explanation of the result parameters can be referred to in [2.2.2 Python Script Integration](#222-python脚本方式集成).

The visualization results are saved under `save_path`, and the visualization result is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/02.png">

#### 2.2.2 Python Script Integration

* The above command line is for quickly experiencing and viewing the effect. Generally, in a project, it is often necessary to integrate through code. You can complete the quick inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="image_multilabel_classification")

output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

The result obtained is the same as the command line method.

In the above Python script, the following steps are performed:

(1) Instantiate the general image multi-label classification pipeline object through `create_pipeline()`, with specific parameter descriptions as follows:

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
<td>Pipeline name or pipeline configuration file path. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
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
<td>Pipeline inference device. Supports specifying the specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the general image multi-label classification pipeline object for inference prediction. This method will return a `generator`. The parameters of the `predict()` method and their descriptions are as follows:

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
  <li><b>str</b>: Local path of an image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as a network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code></li>
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
<tr>
<td><code>threshold</code></td>
<td>Multi-label classification threshold</td>
<td><code>float | dict | list | None</code></td>
<td>
<ul>
  <li><b>float</b>: Represents a unified threshold for multi-label classification;</li>
  <li><b>list</b>: Such as <code>[0.5, 0.45, 0.48, 0.4]</code>, representing different category thresholds in the order of the <code>label list</code>;</li>
  <li><b>dict</b>: Dictionary keys are <code>default</code> and <code>int</code> types, values are <code>float</code> type thresholds, such as <code>{"default": 0.5, 0: 0.45, 2: 0.48, 7: 0.4}</code>, <code>default</code> represents a unified threshold for multi-label classification, other <code>int</code> types apply thresholds of 0.45 for cls_id 0, 0.48 for cls_id 1, and 0.4 for cls_id 7;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.7</code>;</li>
</ul>
</td>
<td><code>0.5</code></td>
</tr>
</table>

3) Process the prediction results. The prediction result for each sample is of type `dict` and supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a json file</td>
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

    - `input_path`: `(str)` Input path of the image to be predicted.
    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.
    - `class_ids`: `(List[numpy.ndarray])` Indicates the class IDs of the prediction results.
    - `scores`: `(List[numpy.ndarray])` Indicates the confidence scores of the prediction results.
    - `label_names`: `(List[str])` Indicates the class names of the prediction results.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}.json`. If specified as a file, it will be saved directly to that file. Since json files do not support saving numpy arrays, `numpy.array` types will be converted to lists.
- Calling the `save_to_img()` method will save the visualization result to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If specified as a file, it will be saved directly to that file. (The pipeline usually contains many result images, it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, leaving only the last image)

* In addition, you can also obtain the visualized image and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is a dictionary type, and the content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type. The key is `res` and the corresponding value is an `Image.Image` object: an object used to display the visualized image of the attribute recognition result.

In addition, you can obtain the general image multi-label classification pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config image_multilabel_classification --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the image multi-label classification pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the configuration file. An example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/image_multilabel_classification.yaml")

output = pipeline.predict(
    input="./general_image_classification_001.jpg",
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are the initialization parameters of the pipeline. If you want to change the initialization parameters of the general image multi-label classification pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. At the same time, CLI prediction also supports passing in the configuration file, and you can specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment directly.

If you need to apply the pipeline directly in your Python project, you can refer to the example code in [2.2 Python Script Integration](#22-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. Therefore, PaddleX provides a high-performance inference plugin, aiming to deeply optimize the performance of model inference and pre/post-processing, achieving significant acceleration of the end-to-end process. For detailed high-performance inference processes, please refer to [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment processes, please refer to [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below is the API reference for basic service deployment and multi-language service call examples:

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
<td>Error code. Fixed at <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed at <code>"Success"</code>.</td>
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
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform image classification.</p>
<p><code>POST /multilabel-image-classification</code></p>
<ul>
<li>The properties of the request body are as follows:</li>
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
<td>The URL of an image file accessible by the server or the Base64-encoded content of the image file.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
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
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>categories</code></td>
<td><code>array</code></td>
<td>Image category information.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code>| <code>null</code></td>
<td>The image classification result. The image is in JPEG format and is encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>categories</code> is an <code>object</code> with the following properties:</p>
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
<td><code>id</code></td>
<td><code>integer</code></td>
<td>Category ID.</td>
</tr>
<tr>
<td><code>name</code></td>
<td><code>string</code></td>
<td>Category name.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Category score.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;categories&quot;: [
{
&quot;id&quot;: 5,
&quot;name&quot;: &quot;Rabbit&quot;,
&quot;score&quot;: 0.93
}
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-Language Service Call Examples</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot; # Service URL
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

# Encode the local image in Base64
with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the returned data
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nCategories:&quot;)
print(result[&quot;categories&quot;])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // <url>https://github.com/Huiyicc/cpp-httplib</url>
#include &quot;nlohmann/json.hpp&quot; // <url>https://github.com/nlohmann/json</url>
#include &quot;base64.hpp&quot; // <url>https://github.com/tobiaslocker/base64</url>

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // Encode the local image in Base64
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
    auto response = client.Post(&quot;/multilabel-image-classification&quot;, headers, body, &quot;application/json&quot;);
    // Process the returned data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outputImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outputImagePath &lt;&lt; std::endl;
        }

        auto categories = result[&quot;categories&quot;];
        std::cout &lt;&lt; &quot;\nCategories:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : categories) {
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
        String API_URL = "http://localhost:8080/multilabel-image-classification"; // Service URL
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
                JsonNode categories = result.get("categories");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                System.out.println("\nCategories: " + categories.toString());
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
    API_URL := "http://localhost:8080/multilabel-image-classification"
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
            Image      string   `json:"image"`
            Categories []map[string]interface{} `json:"categories"`
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
    fmt.Println("\nCategories:")
    for _, category := range respData.Result.Categories {
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
    static readonly string API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local image to Base64
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } }; // Base64-encoded file content or image URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the response data
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nCategories:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;categories&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/multilabel-image-classification'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

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
.then((response) =&gt; {
    // Process the returned data
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log(&quot;\nCategories:&quot;);
    console.log(result[&quot;categories&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;; // Service URL
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

// Encode the local image in Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data); // Base64-encoded file content or image URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the returned data
$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method where computation and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment processes, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate method to deploy the model pipeline based on your needs for subsequent AI application integration.

## 4. Secondary Development
If the default model weights provided by the general image multi-label classification pipeline do not meet your accuracy or speed requirements in your scenario, you can try further <b>fine-tuning</b> the existing model using <b>your own specific domain or application scenario data</b> to improve the recognition performance of the general image multi-label classification pipeline in your scenario.

### 4.1 Model Fine-Tuning

Since the general image multi-label classification pipeline includes an image multi-label classification module, if the performance of the model pipeline is not as expected, you need to refer to the fine-tuning tutorial link in the table below to fine-tune the model.

<table>
  <thead>
    <tr>
      <th>Situation</th>
      <th>Fine-Tuning Module</th>
      <th>Fine-Tuning Reference Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Multi-label classification is inaccurate</td>
      <td>Multi-label classification module</td>
      <td><a href="../../../module_usage/tutorials/cv_modules/image_multilabel_classification.en.md">Link</a></td>
    </tr>
  </tbody>
</table>

### 4.2 Model Application
After completing the fine-tuning training with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights in the corresponding position in the configuration file:

``` yaml
pipeline_name: image_multilabel_classification

SubModules:
  ImageMultiLabelClassification:
    module_name: image_multilabel_classification
    model_name: PP-HGNetV2-B6_ML
    model_dir: null # Modify this path to the local fine-tuned model weight file
    batch_size: 4
```

Subsequently, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

##  5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>You only need to modify the `--device` parameter</b> to achieve seamless switching between different hardware.

For example, if you use Ascend NPU for inference of the general image multi-label classification pipeline, the Python command used is:

```bash
paddlex --pipeline image_multilabel_classification \
        --input general_image_classification_001.jpg \
        --device npu:0
```

If you want to use the general image multi-label classification pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
