---
comments: true
---

# Formula Recognition Pipeline Tutorial

## 1. Introduction to the Formula Recognition Pipeline

Formula recognition is a technology that automatically identifies and extracts LaTeX formula content and its structure from documents or images. It is widely used in document editing and data analysis in fields such as mathematics, physics, and computer science. Leveraging computer vision and machine learning algorithms, formula recognition converts complex mathematical formula information into editable LaTeX format, facilitating further data processing and analysis for users.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/01.jpg">

<b>The Formula Recognition Pipeline comprises a layout detection module and a formula recognition module.</b>

<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model size, choose a model with a smaller storage footprint.</b>


<p><b>Layout Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>92.6</td>
<td>115.126</td>
<td>3827.25</td>
<td>470.2M</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are evaluated on PaddleX's self-built layout detection dataset, containing 10,000 images. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Formula Recognition Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>BLEU Score</th>
<th>Normed Edit Distance</th>
<th>ExpRate (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size</th>
</tr>
</thead>
<tbody>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Trained Model</a></td>
<td>0.8821</td>
<td>0.0823</td>
<td>40.01</td>
<td>-</td>
<td>-</td>
<td>89.7 M</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are measured on the <a href="https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO">LaTeX-OCR Formula Recognition Test Set</a>. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
PaddleX supports experiencing the effects of the formula recognition pipeline through command line or Python locally.

Before using the formula recognition pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.1 Experience via Command Line
Experience the formula recognition pipeline with a single command, using the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png), and replace `--input` with your local path for prediction:

```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device gpu:0
```

Parameter Explanation:

```
--pipeline: The pipeline name, which is formula_recognition for this case.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). Alternatively, use CPU (--device cpu).
```

When executing the above command, the default formula recognition pipeline configuration file is loaded. If you need to customize the configuration file, you can run the following command to obtain it:

<details><summary> 👉Click to Expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config formula_recognition
</code></pre>
<p>After execution, the formula recognition pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can run the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config formula_recognition --save_path ./my_path
</code></pre>
<p>After obtaining the Pipeline configuration file, replace <code>--pipeline</code> with the configuration file's save path to make the configuration file effective. For example, if the configuration file is saved as  <code>./formula_recognition.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./formula_recognition.yaml --input general_formula_recognition.png --device gpu:0
</code></pre>
<p>Here, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified, as they will use the parameters in the configuration file. If parameters are still specified, the specified parameters will take precedence.</p></details>

After execution, the result is:
<details><summary> 👉Click to Expand</summary>

<pre><code>{'input_path': 'general_formula_recognition.png', 'layout_result': {'input_path': 'general_formula_recognition.png', 'boxes': [{'cls_id': 3, 'label': 'number', 'score': 0.7580855488777161, 'coordinate': [1028.3635, 205.46213, 1038.953, 222.99033]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8882032632827759, 'coordinate': [272.75305, 204.50894, 433.7473, 226.17996]}, {'cls_id': 2, 'label': 'text', 'score': 0.9685840606689453, 'coordinate': [272.75928, 282.17773, 1041.9316, 374.44687]}, {'cls_id': 2, 'label': 'text', 'score': 0.9559416770935059, 'coordinate': [272.39056, 385.54114, 1044.1521, 443.8598]}, {'cls_id': 2, 'label': 'text', 'score': 0.9610629081726074, 'coordinate': [272.40817, 467.2738, 1045.1033, 563.4855]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8916195034980774, 'coordinate': [503.45743, 594.6236, 1040.6804, 619.73895]}, {'cls_id': 2, 'label': 'text', 'score': 0.973675549030304, 'coordinate': [272.32007, 648.8599, 1040.8702, 775.15686]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9038916230201721, 'coordinate': [554.2307, 803.5825, 1040.4657, 855.3159]}, {'cls_id': 2, 'label': 'text', 'score': 0.9025381803512573, 'coordinate': [272.535, 875.1402, 573.1086, 898.3587]}, {'cls_id': 2, 'label': 'text', 'score': 0.8336610794067383, 'coordinate': [317.48013, 909.60864, 966.8498, 933.7868]}, {'cls_id': 2, 'label': 'text', 'score': 0.8779091238975525, 'coordinate': [19.704018, 653.322, 72.433235, 1215.1992]}, {'cls_id': 2, 'label': 'text', 'score': 0.8832409977912903, 'coordinate': [272.13028, 958.50806, 1039.7928, 1019.476]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9088466167449951, 'coordinate': [517.1226, 1042.3978, 1040.2208, 1095.7457]}, {'cls_id': 2, 'label': 'text', 'score': 0.9587949514389038, 'coordinate': [272.03336, 1112.9269, 1041.0201, 1206.8417]}, {'cls_id': 2, 'label': 'text', 'score': 0.8885666131973267, 'coordinate': [271.7495, 1231.8752, 710.44495, 1255.7981]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8907185196876526, 'coordinate': [581.2295, 1287.4525, 1039.8014, 1312.772]}, {'cls_id': 2, 'label': 'text', 'score': 0.9559596180915833, 'coordinate': [273.1827, 1341.421, 1041.0299, 1401.7255]}, {'cls_id': 2, 'label': 'text', 'score': 0.875311553478241, 'coordinate': [272.8338, 1427.3711, 789.7108, 1451.1359]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9152213931083679, 'coordinate': [524.9582, 1474.8136, 1041.6333, 1530.7142]}, {'cls_id': 2, 'label': 'text', 'score': 0.9584835767745972, 'coordinate': [272.81665, 1549.524, 1042.9962, 1608.7157]}]}, 'ocr_result': {}, 'table_result': [], 'dt_polys': [array([[ 503.45743,  594.6236 ],
       [1040.6804 ,  594.6236 ],
       [1040.6804 ,  619.73895],
       [ 503.45743,  619.73895]], dtype=float32), array([[ 554.2307,  803.5825],
       [1040.4657,  803.5825],
       [1040.4657,  855.3159],
       [ 554.2307,  855.3159]], dtype=float32), array([[ 517.1226, 1042.3978],
       [1040.2208, 1042.3978],
       [1040.2208, 1095.7457],
       [ 517.1226, 1095.7457]], dtype=float32), array([[ 581.2295, 1287.4525],
       [1039.8014, 1287.4525],
       [1039.8014, 1312.772 ],
       [ 581.2295, 1312.772 ]], dtype=float32), array([[ 524.9582, 1474.8136],
       [1041.6333, 1474.8136],
       [1041.6333, 1530.7142],
       [ 524.9582, 1530.7142]], dtype=float32)], 'rec_formula': ['F({\bf x})=C(F_{1}(x_{1}),\cdot\cdot\cdot,F_{N}(x_{N})).\qquad\qquad\qquad(1)', 'p(\mathbf{x})=c(\mathbf{u})\prod_{i}p(x_{i}).\qquad\qquad\qquad\qquad\qquad\quad\quad~~\quad~~~~~~~~~~~~~~~(2)', 'H_{c}({\bf x})=-\int_{{\bf{u}}}c({\bf{u}})\log c({\bf{u}})d{\bf{u}}.~~~~~~~~~~~~~~~~~~~~~(3)', 'I({\bf x})=-H_{c}({\bf x}).\qquad\qquad\qquad\qquad(4)', 'H({\bf x})=\sum_{i}H(x_{i})+H_{c}({\bf x}).\eqno\qquad\qquad\qquad(5)']}
</code></pre>
<p>Where <code>dt_polys</code> represents the coordinates of the detected formula area, and <code>rec_formula</code> is the detected formula.</p></details>

The visualization result is as follows:
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/02.jpg">

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path. Formula recognition visualization requires a separate environment configuration. Please refer to [2.3 Formula Recognition Pipeline Visualization](#23-formula-recognition-pipeline-visualization) to install the LaTeX rendering engine.


#### 2.2 Python Script Integration
* Quickly perform inference on the pipeline with just a few lines of code, taking the formula recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
```
> ❗ The results obtained from running the Python script are the same as those from the command line.

The Python script above executes the following steps:

（1）Instantiate the formula recognition pipeline object using `create_pipeline`: Specific parameter descriptions are as follows:

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
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available if the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）Invoke the `predict` method of the formula recognition pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

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
<td>Supports passing in the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png">Example</a>.</td>
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
<td>print</td>
<td>Prints results to the terminal</td>
<td><code>- format_json</code>: bool, whether to format the output content with json indentation, default is True;<br/><code>- indent</code>: int, json formatting setting, only valid when format_json is True, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, only valid when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Saves results as a json file</td>
<td><code>- save_path</code>: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br/><code>- indent</code>: int, json formatting setting, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, default is False;</td>
</tr>
<tr>
<td>save_to_img</td>
<td>Saves results as an image file</td>
<td><code>- save_path</code>: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;</td>
</tr>
</tbody>
</table>
If you have a configuration file, you can customize the configurations of the formula recognition pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/formula_recognition.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/formula_recognition.yaml")
output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
```

### 2.3 Formula Recognition Pipeline Visualization
If you need to visualize the formula recognition pipeline, you need to run the following command to install the LaTeX rendering environment:
```python
apt-get install sudo
sudo apt-get update
sudo apt-get install texlive
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-latex-extra
```
After that, use the `save_to_img` method to save the visualization image. The specific command is as follows:
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
    res.save_to_img("./output/")
```
<b>Note</b>: Since the formula recognition visualization process requires rendering each formula image, it may take a relatively long time. Please be patient.

## 3. Development Integration/Deployment
If the formula recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the formula recognition pipeline directly in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end speedups. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

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
<p>Primary operations provided by the service:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Obtain formula recognition results from an image.</p>
<p><code>POST /formula-recognition</code></p>
<ul>
<li>Request body properties:</li>
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
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image file or PDF file accessible by the server, or the Base64 encoded result of the content of the above-mentioned file types. For PDF files with more than 10 pages, only the content of the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>File type. <code>0</code> indicates a PDF file, and <code>1</code> indicates an image file. If this property is not present in the request body, the file type will be inferred based on the URL.</td>
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
<td><code>formulaRecResults</code></td>
<td><code>array</code></td>
<td>Formula recognition results. The array length is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>formulaRecResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>formulas</code></td>
<td><code>array</code></td>
<td>Positions and contents of formulas.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code></td>
<td>Input image. The image is in JPEG format and encoded using Base64.</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>Layout area detection result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>formulas</code> is an <code>object</code> with the following properties:</p>
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
<td><code>poly</code></td>
<td><code>array</code></td>
<td>Formula position. Elements in the array are the vertex coordinates of the polygon enclosing the formula.</td>
</tr>
<tr>
<td><code>latex</code></td>
<td><code>string</code></td>
<td>Formula content.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/formula-recognition&quot;
file_path = &quot;./demo.jpg&quot;

with open(file_path, &quot;rb&quot;) as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode(&quot;ascii&quot;)

payload = {&quot;file&quot;: file_data, &quot;fileType&quot;: 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
for i, res in enumerate(result[&quot;formulaRecResults&quot;]):
    print(&quot;Detected formulas:&quot;)
    print(res[&quot;formulas&quot;])
    layout_img_path = f&quot;layout_{i}.jpg&quot;
    with open(layout_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;layoutImage&quot;]))
    print(f&quot;Output image saved at {layout_img_path}&quot;)
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to proceed with subsequent AI application integration.


## 4. Custom Development
If the default model weights provided by the formula recognition pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing models using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the formula recognition pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the formula recognition pipeline consists of two modules (layout detection and formula recognition), unsatisfactory performance may stem from either module.

You can analyze images with poor recognition results. If you find that many formula are undetected (i.e., formula miss detection), it may indicate that the layout detection model needs improvement. You should refer to the [Customization](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md#iv-custom-development) section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md) and use your private dataset to fine-tune the layout detection model. If many recognition errors occur in detected formula (i.e., the recognized formula content does not match the actual formula content), it suggests that the formula recognition model requires further refinement. You should refer to the [Customization](../../../module_usage/tutorials/ocr_modules/formula_recognition.en.md#iv-custom-development) section in the [Formula Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/formula_recognition.en.md) and fine-tune the formula recognition model.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local paths of the fine-tuned model weights to the corresponding positions in the pipeline configuration file:

```bash
......
Pipeline:
  layout_model: RT-DETR-H_layout_17cls # Can be replaced with the local path of the fine-tuned layout detection model
  formula_rec_model: LaTeX_OCR_rec # Can be replaced with the local path of the fine-tuned formula recognition model
  formula_rec_batch_size: 5
  device: "gpu:0"
......
```

Then, refer to the command line method or Python script method in [2. Quick Start](#2-quick-start) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modifying the `--device` parameter</b> allows seamless switching between different hardware.

For example, if you are using an NVIDIA GPU for formula pipeline inference, the Python command would be:

```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device gpu:0
```
Now, if you want to switch the hardware to Ascend NPU, you only need to modify the `--device` in the Python command:


```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device npu:0
```

If you want to use the formula recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
