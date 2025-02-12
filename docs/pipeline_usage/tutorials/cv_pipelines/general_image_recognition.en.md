---
comments: true
---

# General Image Recognition Pipeline Usage Tutorial

## 1. Introduction to the General Image Recognition Pipeline

The General Image Recognition Pipeline aims to solve the problem of open-domain object localization and recognition. Currently, PaddleX's General Image Recognition Pipeline supports PP-ShiTuV2.

PP-ShiTuV2 is a practical general image recognition system mainly composed of three modules: mainbody detection module, image feature module, and vector retrieval module. The system integrates and improves various strategies in multiple aspects, including backbone network, loss function, data augmentation, learning rate scheduling, regularization, pre-trained model, and model pruning and quantization. It optimizes each module and ultimately achieves better performance in multiple application scenarios.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/general_image_recognition/pp_shitu_v2.jpg"/>
<b>The General Image Recognition Pipeline includes the mainbody detection module and the image feature module</b>, with several models to choose. You can select the model to use based on the benchmark data below. <b>If you prioritize model precision, choose a model with higher precision. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size</b>.


<b>Mainbody Detection Module:</b>
<table>
<tr>
<th>Model</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-ShiTuV2_det</td>
<td>41.5</td>
<td>62.0</td>
<td>12.79 / 4.51</td>
<td>44.14 / 44.14</td>
<td>27.54</td>
<td>An mainbody detection model based on PicoDet_LCNet_x2_5, which may detect multiple common objects simultaneously.</td>
</tr>
</table>

Note: The above accuracy metrics are based on the private mainbody detection dataset.

<b>Image Feature Module:</b>
<table>
<tr>
<th>Model</th>
<th>Recall@1 (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-ShiTuV2_rec</td>
<td>84.2</td>
<td>3.48 / 0.55</td>
<td>8.04 / 4.04</td>
<td>16.3 M</td>
<td rowspan="3">PP-ShiTuV2 is a general image feature system consisting of three modules: mainbody detection, feature extraction, and vector retrieval. These models are part of the feature extraction module, and different models can be selected based on system requirements.</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td>88.69</td>
<td>12.94 / 2.88</td>
<td>58.36 / 58.36</td>
<td>306.6 M</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td>
<td>91.03</td>
<td>51.65 / 11.18</td>
<td>255.78 / 255.78</td>
<td>1.05 G</td>
</tr>
</table>

Note: The above accuracy metrics are based on AliProducts Recall@1. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.

## 2. Quick Start

The pre-trained model pipelines provided by PaddleX can be quickly experienced. You can use Python to experience locally.

### 2.1 Online Experience

Not supported yet.

### 2.2 Local Experience

> ❗ Before using the general image recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience

The pipeline currently does not support command line experience.

#### 2.2.2 Python Script Integration

* To run the pipeline, you need to build an index library in advance. You can download the official beverage recognition test dataset [drink_dataset_v2.0]( https://paddle-model-ecology.bj.bcebos.com/paddlex/data/drink_dataset_v2.0.tar) to build the index library. If you wish to use your private dataset, please refer to [Section 2.3 Data Organization for Building the Index Library](#23-data-organization-for-building-the-index-library). After that, you can quickly build the index library and perform fast inference with the general image recognition pipeline using just a few lines of code.

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ShiTuV2")

index_data = pipeline.build_index(gallery_imgs="drink_dataset_v2.0/", gallery_label="drink_dataset_v2.0/gallery.txt")
index_data.save("drink_index")

output = pipeline.predict("./drink_dataset_v2.0/test_images/001.jpeg", index=index_data)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

In the above Python script, the following steps are executed:

(1) Call the `create_pipeline` to instantiate the general image recognition production line object. The specific parameter descriptions are as follows:

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
<td><code>pipeline</code></td>
<td>The name of the production line or the path to the production line configuration file. If it is a production line name, it must be a production line supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
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
<td>The inference device for the production line. Supports specifying specific GPU card numbers, such as "gpu:0", specific card numbers for other hardware, such as "npu:0", or CPU, such as "cpu".</td>
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

(2) Call the `build_index` method of the general image recognition production line object to build the index library. The specific parameter descriptions are as follows:

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
<tbody>
<tr>
<td><code>gallery_imgs</code></td>
<td>The gallery images to be added. This is a required parameter.</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>: The root directory of the dataset. The data organization method is referenced in <a href="#2.3-Data Organization for Index Building">Section 2.3 Data Organization for Index Building</a>.</li>
<li><b>List[numpy.ndarray]</b>: Gallery image data in the form of a list of numpy arrays.</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>The annotation information of the gallery images. This is a required parameter.</td>
<td><code>str|list</code></td>
<td>
<ul>
<li><b>str</b>: The path to the annotation file. The data organization method is referenced in <a href="#2.3-Data Organization for Index Building">Section 2.3 Data Organization for Index Building</a>.</li>
<li><b>List[str]</b>: Gallery image annotations in the form of a list of strings.</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>The feature measurement method. This is an optional parameter.</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"IP"</code>: Inner Product</li>
<li><code>"L2"</code>: Euclidean Distance</li>
</ul>
</td>
<td><code>"IP"</code></td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>The type of index. This is an optional parameter.</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"HNSW32"</code>: Faster search speed and higher accuracy, but does not support the <code>remove_index()</code> operation.</li>
<li><code>"IVF"</code>: Faster search speed but relatively lower accuracy, supports <code>append_index()</code> and <code>remove_index()</code> operations.</li>
<li><code>"Flat"</code>: Slower search speed but higher accuracy, supports <code>append_index()</code> and <code>remove_index()</code> operations.</li>
</ul>
</td>
<td><code>"HNSW32"</code></td>
</tr>
</tbody>
</table>

The index library object `index` supports the `save` method, which is used to save the index library to disk:

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
<td><code>save_path</code></td>
<td>The directory where the index library file is saved, such as <code>drink_index</code>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>

(3) Call the `predict` method of the general image recognition production line object for inference prediction: The `predict` method takes `input` as a parameter, which is used to input the data to be predicted and supports multiple input methods. Specific examples are as follows:

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
<td>Data to be predicted, supports multiple input types, required parameter</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of the image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of the image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">Example</a>; <b>Local directory</b>, the directory should contain images to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>index</code></td>
<td>The feature library used for production line inference prediction, optional parameter. If this parameter is not provided, the index library specified in the production line configuration file will be used by default.</td>
<td><code>str|paddlex.inference.components.retrieval.faiss.IndexData|None</code></td>
<td>
<ul>
<li><b>str</b> type representing a directory (the directory should contain the feature library files, including <code>vector.index</code> and <code>index_info.yaml</code>)</li>
<li><b>IndexData</b> object created by the <code>build_index</code> method</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

(4) Process the prediction results: The prediction result of each sample is of `dict` type, and it supports printing or saving as a file. The supported file types depend on the specific pipeline, such as:

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
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
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

- Calling the `print()` method will print the following result to the terminal:

```bash
{'res': {'input_path': './drink_dataset_v2.0/test_images/001.jpeg', 'boxes': [{'labels': ['红牛-强化型', '红牛-强化型', '红牛-强化型', '红牛-强化型', '红牛-强化型'], 'rec_scores': [0.720183789730072, 0.7044230699539185, 0.6812724471092224, 0.6583285927772522, 0.6578206419944763], 'det_score': 0.6135568618774414, 'coordinate': [343.8184, 98.96374, 528.0366, 593.3813]}]}}
```

- The meanings of the output parameters are as follows:
    - `input_path`: Indicates the path of the input image
    - `boxes`: Information of detected objects, a list of dictionaries, each dictionary contains the following information:
        - `labels`: List of recognized labels, sorted by score from high to low
        - `rec_scores`: List of recognition scores, where elements correspond to `labels` one by one
        - `det_score`: Detection score
        - `coordinate`: Coordinates of the target box, in the format [xmin, ymin, xmax, ymax]

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}.json`. If a file is specified, it will be saved directly to that file.
- Calling the `save_to_img()` method will save the visualization result to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The production line usually contains many result images, it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, leaving only the last one). In the above example, the visualization result is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/general_image_recognition/01.jpg"/>

* Additionally, it also supports obtaining the visualized image with results and prediction results through attributes, as follows:

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

- The prediction result obtained by the `json` attribute is data of dict type, and the relevant content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of dict type. The key is `res`, and the corresponding value is an `Image.Image` object used to visualize the general image recognition result.

The above Python script integration method uses the parameter settings in the PaddleX official configuration file by default. If you need to customize the configuration file, you can first execute the following command to obtain the official configuration file and save it in `my_path`:

```bash
paddlex --get_pipeline_config PP-ShiTuV2 --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the general image recognition production line. You just need to modify the `pipeline` parameter value in the `create_pipeline` method to the path of your custom production line configuration file.

For example, if your custom configuration file is saved in `./my_path/PP-ShiTuV2.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/PP-ShiTuV2.yaml")

output = pipeline.predict("./drink_dataset_v2.0/test_images/001.jpeg", index="drink_index")
for res in output:
    res.print()
    res.save_to_json("./output/")
    res.save_to_img("./output/")
```

<b>Note:</b> The parameters in the configuration file are the initialization parameters of the pipeline. If you wish to change the initialization parameters of the general image recognition pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction.

#### 2.2.3 Adding and Deleting Operations in the Index Library

If you wish to add more images to the index library, you can call the `append_index` method; to delete image features, you can call the `remove_index` method.

```python
from paddlex import create_pipeline

pipeline = create_pipeline("PP-ShiTuV2")
index_data = pipeline.build_index(gallery_imgs="drink_dataset_v2.0/", gallery_label="drink_dataset_v2.0/gallery.txt", index_type="IVF", metric_type="IP")
index_data = pipeline.append_index(gallery_imgs="drink_dataset_v2.0/", gallery_label="drink_dataset_v2.0/gallery.txt", index=index_data)
index_data = pipeline.remove_index(remove_ids="drink_dataset_v2.0/remove_ids.txt", index=index_data)
index_data.save("drink_index")
```

The parameters of the above method are described as follows:
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
<tbody>
<tr>
<td><code>gallery_imgs</code></td>
<td>Gallery images to be added, required parameter</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>: Root directory of images, data organization refers to <a href="#2.3-Data Organization for Building the Index Library">Section 2.3 Data Organization for Building the Index Library</a></li>
<li><b>List[numpy.ndarray]</b>: Gallery image data in the form of a list of numpy arrays</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>Labels for gallery images, required parameter</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>: Path to the label file, data organization is the same as when building the feature library, refer to <a href="#2.3-Data Organization for Building the Index Library">Section 2.3 Data Organization for Building the Index Library</a></li>
<li><b>List[str]</b>: Gallery image labels in the form of a list of strings</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>Feature measurement method, optional parameter</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"IP"</code>: Inner Product</li>
<li><code>"L2"</code>: Euclidean Distance</li>
</ul>
</td>
<td><code>"IP"</code></td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>Type of index, optional parameter</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"HNSW32"</code>: Faster search speed and higher accuracy, but does not support <code>remove_index()</code> operation</li>
<li><code>"IVF"</code>: Faster search speed but relatively lower accuracy, supports <code>append_index()</code> and <code>remove_index()</code> operations</li>
<li><code>"Flat"</code>: Slower search speed but higher accuracy, supports <code>append_index()</code> and <code>remove_index()</code> operations</li>
</ul>
</td>
<td><code>"HNSW32"</code></td>
</tr>
<tr>
<td><code>remove_ids</code></td>
<td>Indices to be removed</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>: Path to a txt file containing the indices to be removed, one "id" per line;</li>
<li><b>List[int]</b>: List of indices to be removed. Only valid in <code>remove_index</code>.</li></ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>index</code></td>
<td>Feature library used for pipeline inference</td>
<td><code>str|paddlex.inference.components.retrieval.faiss.IndexData</code></td>
<td>
<ul>
<li><b>str</b>: Directory (the directory should contain feature library files, including <code>vector.index</code> and <code>index_info.yaml</code>)</li>
<li><b>IndexData</b> object created by <code>build_index</code> method</li>
</ul>
</td>
<td>None</td>
</tr>
</tbody>
</table>
<b>Note</b>: <code>HNSW32</code> has compatibility issues on the Windows platform, which may prevent the index library from being built or loaded.

### 2.3 Data Organization for Building the Index Library

The general image recognition pipeline example of PaddleX requires a pre-built index library for feature retrieval. If you wish to build an index library with your private data, you need to organize the data as follows:

```bash
data_root             # 数据集根目录，目录名称可以改变
├── images            # 图像的保存目录，目录名称可以改变
│   │   ...
└── gallery.txt       # 索引库数据集标注文件，文件名称可以改变。每行给出待检索图像路径和图像标签，使用空格分隔，内容举例： “0/0.jpg 脉动”
```

## 3. Development Integration/Deployment

If the general image recognition production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the general image recognition production line directly in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX offers a high-performance inference plugin aimed at deeply optimizing the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference processes, please refer to [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating the inference function as a service, clients can access these services via network requests to obtain inference results. PaddleX supports multiple production line service deployment schemes. For detailed production line service deployment processes, please refer to [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

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
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>buildIndex</code></b></li>
</ul>
<p>Build feature vector index.</p>
<p><code>POST /shitu-index-build</code></p>
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
<td><code>imageLabelPairs</code></td>
<td><code>array</code></td>
<td>Image-label pairs used to build the index.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<p>Each element in <code>imageLabelPairs</code> is an <code>object</code> with the following properties:</p>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of the image file accessible by the server or the Base64 encoded result of the image file content.</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>Label.</td>
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
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>The key corresponding to the index, used to identify the created index. It can be used as input for other operations.</td>
</tr>
<tr>
<td><code>idMap</code></td>
<td><code>object</code></td>
<td>Mapping from vector IDs to labels.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>addImagesToIndex</code></b></li>
</ul>
<p>Add images (corresponding feature vectors) to the index.</p>
<p><code>POST /shitu-index-add</code></p>
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
<td><code>imageLabelPairs</code></td>
<td><code>array</code></td>
<td>Image-label pairs used to build the index.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>The key corresponding to the index. Provided by the <code>buildIndex</code> operation.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>Each element in <code>imageLabelPairs</code> is an <code>object</code> with the following properties:</p>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of an image file accessible by the server or the Base64-encoded content of the image file.</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>The label.</td>
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
<td><code>idMap</code></td>
<td><code>object</code></td>
<td>Mapping from vector IDs to labels.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>removeImagesFromIndex</code></b></li>
</ul>
<p>Remove images (corresponding feature vectors) from the index.</p>
<p><code>POST /shitu-index-remove</code></p>
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
<td><code>ids</code></td>
<td><code>array</code></td>
<td>The IDs of the vectors to be removed from the index.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>The key corresponding to the index. Provided by the <code>buildIndex</code> operation.</td>
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
<td><code>idMap</code></td>
<td><code>object</code></td>
<td>Mapping from vector IDs to labels.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform image recognition.</p>
<p><code>POST /shitu-infer</code></p>
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
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>The key corresponding to the index. Provided by the <code>buildIndex</code> operation.</td>
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
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>Information about detected objects.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The recognition result image. The image is in JPEG format and is Base64-encoded.</td>
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
<td>The location of the object. The elements of the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.</td>
</tr>
<tr>
<td><code>recResults</code></td>
<td><code>array</code></td>
<td>Recognition results.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The detection score.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>recResults</code> is an <code>object</code> with the following properties:</p>
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
<td>The label.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The recognition score.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Call Example</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import pprint
import sys

import requests

API_BASE_URL = "http://0.0.0.0:8080"

base_image_label_pairs = [
    {"image": "./demo0.jpg", "label": "Rabbit"},
    {"image": "./demo1.jpg", "label": "Rabbit"},
    {"image": "./demo2.jpg", "label": "Dog"},
]
image_label_pairs_to_add = [
    {"image": "./demo3.jpg", "label": "Dog"},
]
ids_to_remove = [1]
infer_image_path = "./demo4.jpg"
output_image_path = "./out.jpg"

for pair in base_image_label_pairs:
    with open(pair["image"], "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")
    pair["image"] = image_data

payload = {"imageLabelPairs": base_image_label_pairs}
resp_index_build = requests.post(f"{API_BASE_URL}/shitu-index-build", json=payload)
if resp_index_build.status_code != 200:
    print(f"Request to shitu-index-build failed with status code {resp_index_build}.")
    pprint.pp(resp_index_build.json())
    sys.exit(1)
result_index_build = resp_index_build.json()["result"
print(f"Number of images indexed: {len(result_index_build['idMap'])}")

for pair in image_label_pairs_to_add:
    with open(pair["image"], "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")
    pair["image"] = image_data

payload = {"imageLabelPairs": image_label_pairs_to_add, "indexKey": result_index_build["indexKey"]}
resp_index_add = requests.post(f"{API_BASE_URL}/shitu-index-add", json=payload)
if resp_index_add.status_code != 200:
    print(f"Request to shitu-index-add failed with status code {resp_index_add}.")
    pprint.pp(resp_index_add.json())
    sys.exit(1)
result_index_add = resp_index_add.json()["result"]
print(f"Number of images indexed: {len(result_index_add['idMap'])}")

payload = {"ids": ids_to_remove, "indexKey": result_index_build["indexKey"]}
resp_index_remove = requests.post(f"{API_BASE_URL}/shitu-index-remove", json=payload)
if resp_index_remove.status_code != 200:
    print(f"Request to shitu-index-remove failed with status code {resp_index_remove}.")
    pprint.pp(resp_index_remove.json())
    sys.exit(1)
result_index_remove = resp_index_remove.json()["result"]
print(f"Number of images indexed: {len(result_index_remove['idMap'])}")

with open(infer_image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data, "indexKey": result_index_build["indexKey"]}
resp_infer = requests.post(f"{API_BASE_URL}/shitu-infer", json=payload)
if resp_infer.status_code != 200:
    print(f"Request to shitu-infer failed with status code {resp_infer}.")
    pprint.pp(resp_infer.json())
    sys.exit(1)
result_infer = resp_infer.json()["result"]

with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result_infer["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected objects:")
pprint.pp(result_infer["detectedObjects"])
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method where computation and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment processes, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate method to deploy the model production line based on your needs for subsequent AI application integration.

## 4. Secondary Development

If the default model weights provided by the general image recognition production line do not meet your accuracy or speed requirements in your scenario, you can try further <b>fine-tuning</b> the existing model using <b>your own specific domain or application scenario data</b> to improve the recognition performance of the production line in your scenario.

### 4.1 Model Fine-Tuning

Since the general image recognition production line includes two modules (main body detection module and image feature module), the suboptimal performance of the model production line may come from either module.

You can analyze the images with poor recognition results. If you find that many main body targets are not detected during the analysis, it may be due to the inadequacy of the main body detection model. You need to refer to the [Main Body Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/mainbody_detection.en.md) in the [Secondary Development](../../../module_usage/tutorials/cv_modules/mainbody_detection.en.md) section to fine-tune the main body detection model using your private dataset. If there are matching errors in the detected main bodies, it indicates that the image feature model needs further improvement. You need to refer to the [Image Feature Module Development Tutorial](../../../module_usage/tutorials/cv_modules/image_feature.en.md) in the [Secondary Development](../../../module_usage/tutorials/cv_modules/image_feature.en.md) section to fine-tune the image feature model.

### 4.2 Model Application

After completing the fine-tuning training with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the production line configuration file by replacing the local path of the fine-tuned model weights in the corresponding position in the configuration file:

```yaml

...

SubModules:
  Detection:
    module_name: text_detection
    model_name: PP-ShiTuV2_det
    model_dir: null #可修改为微调后主体检测模型的本地路径
    batch_size: 1
  Recognition:
    module_name: text_recognition
    model_name: PP-ShiTuV2_rec
    model_dir: null #可修改为微调后图像特征模型的本地路径
    batch_size: 1
```

Subsequently, refer to the command line method or Python script method in [2.2 Local Experience](#22-本地体验) to load the modified production line configuration file.

##  5. Multi-Hardware Support

PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>You only need to modify the `--device` parameter</b> to achieve seamless switching between different hardware.

For example, when running the general image recognition production line using Python, to change the running device from NVIDIA GPU to Ascend NPU, you only need to modify the `device` in the script to npu:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="PP-ShiTuV2",
    device="npu:0" # gpu:0 --> npu:0
    )
```

If you want to use the general image recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
