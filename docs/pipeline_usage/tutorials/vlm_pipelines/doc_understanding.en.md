# Document Understanding Pipeline User Guide

## 1. Introduction to Document Understanding Pipeline

The Document Understanding Pipeline is an advanced document processing technology based on Vision-Language Models (VLM), designed to overcome the limitations of traditional document processing. Traditional methods rely on fixed templates or predefined rules to parse documents, but this pipeline uses the multimodal capabilities of VLM to accurately answer user questions by integrating visual and linguistic information with just the input of document images and user queries. This technology does not require pre-training for specific document formats, allowing it to flexibly handle diverse document content and significantly enhance the generalization and practicality of document processing. It has broad application prospects in scenarios such as intelligent Q&A and information extraction. Currently, this pipeline does not support secondary development of VLM models, but future support is planned.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/doc_understanding/doc_understanding.png">

<b>The Document Understanding Pipeline includes document-based vision-language model modules. You can choose the model to use based on the benchmark test data below.</b>

<b>If you prioritize model accuracy, choose a model with higher accuracy; if you care more about inference speed, choose a faster model; if you are concerned about storage size, choose a model with a smaller storage footprint.</b>

<p><b>Document-based Vision-Language Model Modules (Optional):</b></p>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Model Storage Size (GB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-DocBee-2B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocBee-2B_infer.tar">Inference Model</a></td>
<td>4.2</td>
<td rowspan="2">PP-DocBee is a self-developed multimodal large model by the PaddlePaddle team, focusing on document understanding with excellent performance on Chinese document understanding tasks. The model is fine-tuned with nearly 5 million multimodal datasets for document understanding, including general VQA, OCR, chart, text-rich documents, mathematics and complex reasoning, synthetic data, and pure text data, with different training data ratios. On several authoritative English document understanding evaluation benchmarks in academia, PP-DocBee has achieved SOTA for models of the same parameter scale. In internal business Chinese scenarios, PP-DocBee also outperforms current popular open and closed-source models.</td>
</tr>
<tr>
<td>PP-DocBee-7B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocBee-7B_infer.tar">Inference Model</a></td>
<td>15.8</td>
</tr>
</table>

## 2. Quick Start

### 2.1 Local Experience

> ❗ Before using the Document Understanding Pipeline locally, ensure you have installed the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.md). If you wish to selectively install dependencies, refer to the relevant instructions in the installation guide. The dependency group for this pipeline is `multimodal`.

#### 2.1.1 Integration via Python Script

* The Document Understanding Pipeline can be quickly inferred with just a few lines of code as shown below:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="doc_understanding")
output = pipeline.predict(
    {
        "image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png",
        "query": "Identify the contents of this table"
    }
)
for res in output:
    res.print()
    res.save_to_json("./output/")
```

In the above Python script, the following steps are performed:

1. Instantiate the Document Understanding Pipeline object through `create_pipeline()`. The parameter details are as follows:

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
<td>Pipeline name or configuration file path. If it's a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has a higher priority and requires the pipeline name to be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device for the pipeline. Supports specifying a specific GPU card number, such as "gpu:0", or other hardware card numbers, like "npu:0", or CPU as "cpu".</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin. If set to `None`, the setting from the configuration file will be used. Not supported for now.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration. Not supported for now.</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

2. Call the `predict()` method of the Document Understanding Pipeline object for inference prediction. This method returns a `generator`. Below are the parameters of the `predict()` method and their descriptions:

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
<td>Data to be predicted, currently only supports dictionary-type input</td>
<td><code>Python Dict</code></td>
<td>
<ul>
  <li><b>Python Dict</b>: For PP-DocBee, the input format is: <code>{"image":/path/to/image, "query": user question}</code>, representing the input image and the corresponding user question.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device for the pipeline</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: e.g., <code>cpu</code> for CPU inference;</li>
  <li><b>GPU</b>: e.g., <code>gpu:0</code> for inference on the first GPU;</li>
  <li><b>NPU</b>: e.g., <code>npu:0</code> for inference on the first NPU;</li>
  <li><b>XPU</b>: e.g., <code>xpu:0</code> for inference on the first XPU;</li>
  <li><b>MLU</b>: e.g., <code>mlu:0</code> for inference on the first MLU;</li>
  <li><b>DCU</b>: e.g., <code>dcu:0</code> for inference on the first DCU;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized by the pipeline will be used. During initialization, it will preferentially use the local GPU 0 device if available, otherwise the CPU device will be used;</li>
</ul>
</td>
<td><code>None</code></td>
</table>

3. Process the prediction results. The prediction result for each sample is a corresponding Result object, and supports operations such as printing and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Description</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify the <code>JSON</code> output, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save results as a json format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file name is consistent with the input file type name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify the <code>JSON</code> output, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The printed content includes:

  - `image`: `(str)` The input path of the image

  - `query`: `(str)` The question related to the input image

  - `result`: `(str)` The output result from the model

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.json`. If specified as a file, it will be directly saved to that file.

* Additionally, you can also access the visualized image with results and prediction results through attributes, as follows:

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

- The prediction result obtained from the `json` attribute is data of type dict, and the related content is consistent with the content saved by calling the `save_to_json()` method.

Additionally, you can obtain the configuration file for the Document Understanding Pipeline and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config doc_understanding --save_path ./my_path
```

If you have obtained the configuration file, you can customize various configurations of the Document Understanding Pipeline, just modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. For example:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/doc_understanding.yaml")
output = pipeline.predict(
    {
        "image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png",
        "query": "Identify the contents of this table"
    }
)
for res in output:
    res.print()
    res.save_to_json("./output/")
```

<b>Note:</b> The parameters in the configuration file are the initialization parameters of the pipeline. If you want to change the initialization parameters of the Document Understanding Pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Meanwhile, CLI prediction also supports passing in the configuration file, `--pipeline` specifies the path of the configuration file.

## 3. Development Integration/Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can directly proceed to development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.1.2 Python Script Method](#212-python-script-method-integration).

In addition, PaddleX also provides three other deployment methods, as detailed below:

🚀 <b>High-Performance Inference</b> (This pipeline does not support it currently): In real-world production environments, many applications have strict standards for performance indicators of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deeply optimizing model inference and pre-post processing to achieve significant speed improvements in end-to-end processes. For detailed high-performance inference processes, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various pipeline service deployment solutions. For detailed pipeline service deployment processes, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.md).

Below is a basic service deployment API reference and multilingual service call example:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body attributes are as follows:</li>
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
<li>When the request is not processed successfully, the response body attributes are as follows:</li>
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
<td>Error code. The same as the response status code.</td>
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
<p>Perform object detection on an image.</p>
<p><code>POST /open-vocabulary-detection</code></p>
<ul>
<li>The request body attributes are as follows:</li>
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
<td>A URL of the image file accessible by the server or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>prompt</code></td>
<td><code>string</code></td>
<td>The text prompt used for prediction.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>thresholds</code></td>
<td><code>object</code> | <code>null</code></td>
<td>The thresholds used for model prediction.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the response body's <code>result</code> has the following attributes:</li>
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
<td>Information on the position, category, etc., of the object.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>Result image of object detection. The image is in JPEG format, encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>detectedObjects</code> is an <code>object</code>, with the following attributes:</p>
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
<td>The position of the object. The elements in the array are the x-coordinate of the upper-left corner, the y-coordinate of the upper-left corner, the x-coordinate of the lower-right corner, and the y-coordinate of the lower-right corner respectively.</td>
</tr>
<tr>
<td><code>categoryName</code></td>
<td><code>string</code></td>
<td>The category name of the object.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The score of the object.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
"detectedObjects": [
{
"bbox": [
404.4967956542969,
90.15770721435547,
506.2465515136719,
285.4187316894531
],
"categoryName": "bird",
"score": 0.7418514490127563
},
{
"bbox": [
155.33145141601562,
81.10954284667969,
199.71136474609375,
167.4235382080078
],
"categoryName": "dog",
"score": 0.7328268885612488
}
],
"image": "xxxxxx"
}
</code></pre></details>

<details><summary>Examples of Multilingual Service Calls</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64

</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a way of placing computing and data processing functions on the user device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment processes, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md).
You can choose the appropriate deployment method for your needs and proceed with subsequent AI application integration.


## 4. Secondary Development

Currently, this pipeline does not support fine-tuning training and only supports inference integration. Future support for fine-tuning training is planned.

## 5. Multi-Hardware Support

Currently, this pipeline only supports GPU and CPU inference. Future support for more hardware adaptations is planned.
