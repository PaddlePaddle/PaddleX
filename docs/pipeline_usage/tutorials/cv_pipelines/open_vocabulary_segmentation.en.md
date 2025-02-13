---
comments: true
---

# Open Vocabulary Segmentation Pipeline User Guide

## 1. Introduction to Open Vocabulary Segmentation Pipeline
Open vocabulary segmentation is an image segmentation task that aims to segment objects in an image based on text descriptions, bounding boxes, key points, and other information besides the image itself. It allows the model to handle a wide range of object categories without a predefined category list. This technology combines visual and multimodal techniques, greatly enhancing the flexibility and accuracy of image processing. Open vocabulary segmentation has significant application value in the field of computer vision, especially in object segmentation tasks in complex scenarios. This pipeline also provides flexible service deployment options, supporting multiple programming languages on various hardware. Currently, this pipeline does not support secondary development of the model, but it is planned to be supported in the future.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg">

<b>The general open vocabulary segmentation pipeline includes an open vocabulary segmentation module. You can choose the model based on the benchmark data below.</b>

<b>If you prioritize model accuracy, choose a model with higher accuracy; if you prioritize inference speed, choose a model with faster inference speed; if you prioritize storage size, choose a model with a smaller storage size.</b>

<p><b>General Image Open Vocabulary Segmentation Module (Optional):</b></p>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SAM-H_box</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SAM-H_box_infer.tar">Inference Model</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td rowspan="2">SAM (Segment Anything Model) is an advanced image segmentation model that can segment any object in an image based on simple prompts provided by the user (such as points, boxes, or text). Trained on the SA-1B dataset, which contains ten million images and one billion mask annotations, it performs well in most scenarios. SAM-H_box uses a box as the segmentation prompt input, and SAM will segment the main subject enclosed by the box; SAM-H_point uses a point as the segmentation prompt input, and SAM will segment the subject at the point.</td>
</tr>
<tr>
<td>SAM-H_point</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SAM-H_point_infer.tar">Inference Model</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
</tr>
</table>

<b>Note: All model GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision, and CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## 2. Quick Start

### 2.1 Local Experience
> ❗ Before using the general open vocabulary segmentation pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.1.1 Command Line Experience
* You can quickly experience the open vocabulary segmentation pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg) and replace `--input` with your local path for prediction.

```bash
paddlex --pipeline open_vocabulary_segmentation \
        --input open_vocabulary_segmentation.jpg \
        --prompt_type box \
        --prompt "[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]" \
        --save_path ./output \
        --device gpu:0
```

The relevant parameter description can be found in the parameter description in [2.1.2 Integration via Python Script]().

After running, the result will be printed to the terminal, as follows:

```bash
{'res': {'input_path': 'open_vocabulary_segmentation.jpg', 'prompts': {'box_prompt': [[112.9, 118.4, 513.8, 382.1], [4.6, 263.6, 92.2, 336.6], [592.4, 260.9, 607.2, 294.2]]}, 'masks': '...', 'mask_infos': [{'label': 'box_prompt', 'prompt': [112.9, 118.4, 513.8, 382.1]}, {'label': 'box_prompt', 'prompt': [4.6, 263.6, 92.2, 336.6]}, {'label': 'box_prompt', 'prompt': [592.4, 260.9, 607.2, 294.2]}]}}
```

The explanation of the running result parameters can be found in the result explanation of [2.1.2 Python Script Integration](#212-python-script-integration).

The visualization results are saved under `save_path`, and the visualization result of open vocabulary segmentation is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg">

#### 2.1.2 Python Script Integration
* The above command line is for quickly experiencing the effect. Generally, in a project, integration through code is often required. You can complete the rapid inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline_name="open_vocabulary_segmentation")
output = pipeline.predict(input="open_vocabulary_segmentation.jpg", prompt_type="box", prompt=[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]])
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps are executed:

(1) The <code>create_pipeline()</code> function is used to instantiate an Open Vocabulary Segmentation pipeline object, with the specific parameter descriptions as follows:

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
<td>The inference device for the pipeline. It supports specifying the exact card number for GPU, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu".</td>
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

(2) The <code>predict()</code> method of the Open Vocabulary Segmentation pipeline object is called to perform inference prediction. This method returns a <code>generator</code>. Below are the parameters and their descriptions for the <code>predict()</code> method:

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
<td>The data to be predicted, supporting multiple input types (required).</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code>.</li>
  <li><b>str</b>: Local path of an image file or PDF file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg">example</a>; <b>Local directory</b>, which must contain images to be predicted, such as the local path: <code>/root/data/</code> (currently, prediction of PDF files in directories is not supported; PDF files must be specified with an exact file path).</li>
  <li><b>List</b>: The elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference;</li>
  <li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
  <li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference;</li>
  <li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference;</li>
  <li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference;</li>
  <li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference;</li>
  <li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default. During initialization, the local GPU device 0 will be prioritized; if unavailable, the CPU device will be used.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>prompt_type</code></td>
<td>The type of prompt used during model inference.</td>
<td><code>str</code></td>
<td>
<ul>
    <li><b>box</b>: Use bounding boxes as prompt inputs. If set to <code>box</code>, the input prompt must be in the form of <code>list[list[float, float, float, float]]</code>.</li>
    <li><b>point</b>: Use points as prompt inputs. If set to <code>point</code>, the input prompt must be in the form of <code>list[list[float, float]]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>prompt</code></td>
<td>The specific prompt used during model inference.</td>
<td><code>list[list[float]]</code></td>
<td>
<ul>
    <li><b>list[list[float]]</b>: Must be set according to the type specified by <code>prompt_type</code>.</li>
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

    - `prompts`: `(dict)` The original prompt information used for predicting the image

    - `masks`: `...` The actual predicted masks of the segmentation model. Due to the large data size, it is replaced with `...` for printing. You can save the prediction results as an image using `res.save_to_img` or as a JSON file using `res.save_to_json`.

    - `mask_infos`: `(list)` Segmentation result information corresponding to the elements in `masks`, with the same length as `masks`. Each element is a dictionary containing the following fields
      - `label`: `(str)` The type of prompt used to predict the corresponding element in `masks`, such as `box_prompt` indicating that the corresponding mask was obtained using a bounding box as the prompt
      - `prompt`: `list` The specific prompt information used for predicting the corresponding element in `masks`

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
- The prediction result returned by the `img` attribute is a dictionary type of data. The key is `res`, and the corresponding value is an `Image.Image` object used for visualizing the open vocabulary segmentation results.

In addition, you can obtain the open vocabulary segmentation pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config open_vocabulary_segmentation --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the open vocabulary segmentation pipeline. Simply modify the value of the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file. An example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/open_vocabulary_segmentation.yaml")

output = pipeline.predict(
    input="./open_vocabulary_segmentation.jpg",
    prompt_type="box",
    prompt=[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are for pipeline initialization. If you wish to change the initialization parameters of the general open-vocabulary segmentation pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in the configuration file by specifying the path with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly to your Python project, you can refer to the example code in [2.1.2 Python Script Integration](#212-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. For this purpose, PaddleX provides a high-performance inference plugin, aimed at deeply optimizing the performance of model inference and pre/post-processing, significantly accelerating the end-to-end process. For detailed high-performance inference procedures, please refer to [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating the inference function as a service, clients can access these services via network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references and multi-language service invocation examples for basic service deployment:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body has the following attributes:</li>
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
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the response body has the following attributes:</li>
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
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform object segmentation on an image.</p>
<p><code>POST /open-vocabulary-segmentation</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
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
<td>The URL of the image file accessible by the server or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>prompt</code></td>
<td><code>array</code></td>
<td>The prompt used for prediction.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>promptType</code></td>
<td><code>string</code></td>
<td>The type of prompt used for prediction.</td>
<td>Yes</td>
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
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>masks</code></td>
<td><code>array</code></td>
<td>The segmentation prediction results.</td>
</tr>
<tr>
<td><code>maskInfos</code></td>
<td><code>array</code></td>
<td>Corresponds one-to-one with the elements in the <code>masks</code> field, recording the prompts used for the corresponding segmentation results in <code>masks</code>.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The segmentation result image. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table>
<b>Note</b>: Considering network transmission, the segmentation results recorded in the <code>masks</code> field are encoded with <code>rle</code>. To obtain the original segmentation results, you need to decode them using <code>pycocotools.mask.decode</code>.
</br>
<p>Each element in <code>detectedObjects</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>label</code></td>
<td><code>string</code></td>
<td>The category of the prompt used to generate the mask.</td>
</tr>
<tr>
<td><code>prompt</code></td>
<td><code>array</code></td>
<td>The prompt array.</td>
</tr>
</tbody>
</table>
<p><code>result</code> example is as follows:</p>
<pre><code class="language-python">
{
    'masks': [rle_mask1, rle_mask2, rle_mask3]
    'mask_infos': [
        {'label': 'box_prompt', 'prompt': [112.9, 118.4, 513.8, 382.1]},
        {'label': 'box_prompt', 'prompt': [4.6, 263.6, 92.2, 336.6]},
        {'label': 'box_prompt', 'prompt': [592.4, 260.9, 607.2, 294.2]}
    ]
}
</code>
</details>

<details><summary>Multi-language Service Call Example</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/open-vocabulary-segmentation" # Service URL
image_path = "./open_vocabulary_segmentation.jpg"
output_image_path = "./out.jpg"

# Encode the local image with Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "image": image_data, # Base64-encoded file content or image URL
    "prompt_type": "box",
    "prompt": [[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]
}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the returned data from the API
assert response.status_code == 200
result = response.json()["result"]
image_base64 = result["image"]
image = base64.b64decode(image_base64)
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nresult(with rle encoded binary mask):")
print(result)
</code></pre></details>


</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate method to deploy the model pipeline according to your needs, and then proceed with subsequent AI application integration.

## 4. Secondary Development
The current pipeline temporarily does not support fine-tuning training, only inference integration is supported. Fine-tuning training for this pipeline is planned to be supported in the future.

## 5. Multi-Hardware Support
The current pipeline temporarily only supports GPU and CPU inference. Adaptation to more hardware for this pipeline is planned to be supported in the future.