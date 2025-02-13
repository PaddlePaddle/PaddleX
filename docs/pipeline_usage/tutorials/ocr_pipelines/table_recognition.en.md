---
comments: true
---

# General Table Recognition Pipeline Tutorial

## 1. Introduction to the General Table Recognition Pipeline
Table recognition is a technology that automatically identifies and extracts table content and its structure from documents or images. It is widely used in data entry, information retrieval, and document analysis. By leveraging computer vision and machine learning algorithms, table recognition can convert complex table information into editable formats, facilitating further data processing and analysis for users.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/01.png"/>
<b>The General Table Recognition Pipeline comprises modules for table structure recognition, layout analysis, text detection, and text recognition.</b>
<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model size, choose a model with a smaller storage footprint.</b>
<details><summary> 👉Model List Details</summary>
<p><b>Table Recognition Module Models</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Trained Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td rowspan="1">SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Trained Model</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td rowspan="1">
SLANet_plus is an enhanced version of SLANet, a table structure recognition model developed by Baidu PaddleX Team. Compared to SLANet, SLANet_plus significantly improves its recognition capabilities for wireless and complex tables, while reducing the model's sensitivity to the accuracy of table localization. Even when there are offsets in table localization, it can still perform relatively accurate recognition.
</td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are measured on PaddleX's internal self-built English table recognition dataset. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Layout Analysis Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Trained Model</a></td>
<td>86.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate five types of areas, including text, titles, tables, images, and lists.</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Trained Model</a></td>
<td>95.7</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate one type of tables.</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>87.1</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>An high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>70.3</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>89.3</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>An efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>79.9</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>95.9</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>92.6</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are evaluated on PaddleX's self-built layout analysis dataset containing 10,000 images. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Trained Model</a></td>
<td>77.79</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.2 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Trained Model</a></td>
<td>82.69</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>100.1M</td>
</tr>
</tbody>
</table></details>

## 2. Quick Start
PaddleX's pre-trained model pipelines allow for quick experience of their effects. You can experience the effects of the General Image Classification pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/91661/webUI) the effects of the General Table Recognition pipeline by using the demo images provided by the official. For example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/02.png"/>

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the models in the pipeline online</b>.

### 2.2 Local Experience
Before using the General Table Recognition pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.1 Command Line Experience
Experience the effects of the table recognition pipeline with a single command:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it's the table recognition pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the 1st and 2nd GPUs). CPU can also be selected (--device cpu).
```

When executing the above command, the default table recognition pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> 👉Click to expand</summary>
<pre><code class="language-bash">paddlex --get_pipeline_config table_recognition
</code></pre>
<p>After execution, the table recognition pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config table_recognition --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, replace <code>--pipeline</code> with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is <code>./table_recognition.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./table_recognition.yaml --input table_recognition.jpg --device gpu:0
</code></pre>
<p>Here, parameters like <code>--model</code> and <code>--device</code> do not need to be specified, as they will use the parameters in the configuration file. If they are still specified, the specified parameters will take precedence.</p></details>

After running, the result is:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/03.png"/>

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

### 2.2 Python Script Integration
A few lines of code are all you need to quickly perform inference with the pipeline. Taking the General Table Recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition")

output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the results in img format
    res.save_to_xlsx("./output/")  # Save the results in Excel format
    res.save_to_html("./output/") # Save results in HTML format
```
The results are the same as those obtained through the command line.

In the above Python script, the following steps are executed:

（1）Instantiate the  pipeline object using `create_pipeline`: Specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available if the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）Invoke the `predict` method of the  pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Parameter Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing in Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">Example</a>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td>dict</td>
<td>Supports passing in a dictionary type, where the key needs to correspond to a specific task, such as "img" for image classification tasks. The value of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td>list</td>
<td>Supports passing in a list, where the list elements need to be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
（3）Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

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
<td>save_to_img</td>
<td>Save the results as an img format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
<tr>
<td>save_to_html</td>
<td>Save the results as an html format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
<tr>
<td>save_to_xlsx</td>
<td>Save the results as a spreadsheet format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
</tbody>
</table>
Where `save_to_img` can save visualization results (including OCR result images, layout analysis result images, table structure recognition result images), `save_to_html` can directly save the table as an html file (including text and table formatting), and `save_to_xlsx` can save the table as an Excel format file (including text and formatting).

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/table_recognition.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/table_recognition.yaml")
output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save results in img format
    res.save_to_xlsx("./output/")  # Save results in Excel format
    res.save_to_html("./output/") # Save results in HTML format
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
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
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
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
<p>Locate and recognize tables in the image.</p>
<p><code>POST /table-recognition</code></p>
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
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image file or PDF file accessible by the server, or the Base64 encoding result of the content of the above file types. For PDF files exceeding 10 pages, only the content of the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>The file type. <code>0</code> represents a PDF file, and <code>1</code> represents an image file. If this attribute is not present in the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_doc_orientation_classify</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_doc_unwarping</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_layout_detection</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrModel</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_ocr_model</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>layout_threshold</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>layout_nms</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>See the description of the <code>layout_unclip_ratio</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>See the description of the <code>layout_merge_bboxes_mode</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>See the description of the <code>text_det_limit_side_len</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>See the description of the <code>text_det_limit_type</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_det_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_det_box_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_det_unclip_ratio</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_rec_score_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
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
<td><code>tableRecResults</code></td>
<td><code>object</code></td>
<td>The table recognition result. The length of the array is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>tableRecResults</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>The simplified version of the <code>res</code> field in the JSON representation generated by the <code>predict</code> method of the production object, with the <code>input_path</code> field removed.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>A key-value pair of the input image and the predicted result image. The images are in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multi-language service call example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/table-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["tableRecResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general table recognition pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the general table recognition pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general table recognition pipeline consists of four modules, unsatisfactory performance may stem from any of these modules.

Analyze images with poor recognition results and follow the rules below for analysis and model fine-tuning:

* If the detected table structure is incorrect (e.g., row and column recognition errors, incorrect cell positions), the table structure recognition module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md#customization) section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md) and use your private dataset to fine-tune the table structure recognition model.
* If the table area is incorrectly located within the overall layout, the layout detection module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md#customization) section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md) and use your private dataset to fine-tune the layout detection model.
* If many texts are undetected (i.e., text miss detection), the text detection model may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md#customization) section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md) and use your private dataset to fine-tune the text detection model.
* If many detected texts contain recognition errors (i.e., the recognized text content does not match the actual text content), the text recognition model requires further improvement. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md#customization) section.
### 4.2 Model Application
After fine-tuning your model with a private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the configuration file:

```python
......
 Pipeline:
  layout_model: PicoDet_layout_1x  # Can be modified to the local path of the fine-tuned model
  table_model: SLANet  # Can be modified to the local path of the fine-tuned model
  text_det_model: PP-OCRv4_mobile_det  # Can be modified to the local path of the fine-tuned model
  text_rec_model: PP-OCRv4_mobile_rec  # Can be modified to the local path of the fine-tuned model
  layout_batch_size: 1
  text_rec_batch_size: 1
  table_batch_size: 1
  device: "gpu:0"
......
```
Then, refer to the command line or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for table recognition pipeline inference, the Python command is:

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
At this time, if you want to switch the hardware to Ascend NPU, simply modify `--device` in the Python command to npu:

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device npu:0
```
If you want to use the general table recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
