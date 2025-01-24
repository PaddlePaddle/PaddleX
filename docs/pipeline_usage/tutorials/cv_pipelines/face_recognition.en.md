---
comments: true
---

# Face Recognition Pipeline Tutorial

## 1. Introduction to the Face Recognition Pipeline
Face recognition is a crucial component in the field of computer vision, aiming to automatically identify individuals by analyzing and comparing facial features. This task involves not only detecting faces in images but also extracting and matching facial features to find corresponding identity information in a database. Face recognition is widely used in security authentication, surveillance systems, social media, smart devices, and other scenarios.

The face recognition pipeline is an end-to-end system dedicated to solving face detection and recognition tasks. It can quickly and accurately locate face regions in images, extract facial features, and retrieve and compare them with pre-established features in a feature database to confirm identity information.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/face_recognition/01.png">

<b>The face recognition pipeline includes a face detection module and a face feature module</b>, with several models in each module. Which models to use can be selected based on the benchmark data below. <b>If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model size, choose models with smaller storage requirements</b>.


<p><b>Face Detection Module</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>AP (%)<br/>Easy/Medium/Hard</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>BlazeFace</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace_pretrained.pdparams">Trained Model</a></td>
<td>77.7/73.4/49.5</td>
<td></td>
<td></td>
<td>0.447</td>
<td>A lightweight and efficient face detection model</td>
</tr>
<tr>
<td>BlazeFace-FPN-SSH</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace-FPN-SSH_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace-FPN-SSH_pretrained.pdparams">Trained Model</a></td>
<td>83.2/80.5/60.5</td>
<td>52.4</td>
<td>73.2</td>
<td>0.606</td>
<td>Improved BlazeFace with FPN and SSH structures</td>
</tr>
<tr>
<td>PicoDet_LCNet_x2_5_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">Trained Model</a></td>
<td>93.7/90.7/68.1</td>
<td>33.7</td>
<td>185.1</td>
<td>28.9</td>
<td>Face detection model based on PicoDet_LCNet_x2_5</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_face_pretrained.pdparams">Trained Model</a></td>
<td>93.9/91.8/79.8</td>
<td>25.8</td>
<td>159.9</td>
<td>26.5</td>
<td>Face detection model based on PP-YOLOE_plus-S</td>
</tr>
</tbody>
</table>
<p>Note: The above accuracy metrics are evaluated on the WIDER-FACE validation set with an input size of 640x640. All GPU inference times are based on an NVIDIA V100 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz and FP32 precision.</p>
<p><b>Face Recognition Module</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Output Feature Dimension</th>
<th>Acc (%)<br/>AgeDB-30/CFP-FP/LFW</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileFaceNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileFaceNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileFaceNet_pretrained.pdparams">Trained Model</a></td>
<td>128</td>
<td>96.28/96.71/99.58</td>
<td>5.7</td>
<td>101.6</td>
<td>4.1</td>
<td>Face recognition model trained on MS1Mv3 based on MobileFaceNet</td>
</tr>
<tr>
<td>ResNet50_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_face_pretrained.pdparams">Trained Model</a></td>
<td>512</td>
<td>98.12/98.56/99.77</td>
<td>8.7</td>
<td>200.7</td>
<td>87.2</td>
<td>Face recognition model trained on MS1Mv3 based on ResNet50</td>
</tr>
</tbody>
</table>
<p>Note: The above accuracy metrics are Accuracy scores measured on the AgeDB-30, CFP-FP, and LFW datasets, respectively. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</p>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX can be quickly experienced. You can experience the effects of the face recognition pipeline online or locally using command-line or Python.

### 2.1 Online Experience

Oneline Experience is not supported at the moment.

### 2.2 Local Experience
> ❗ Before using the facial recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.md).

#### 2.2.1 Command Line Experience

Command line experience is not supported at the moment.
#### 2.2.2 Integration via Python Script

Please download the [test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/friends1.jpg) for testing. In the example of running this pipeline, you need to pre-build a facial feature library. You can refer to the following instructions to download the official demo data to be used for subsequent construction of the facial feature library. You can use the following command to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_demo_gallery.tar
tar -xf ./face_demo_gallery.tar
```

If you wish to build a facial feature library using a private dataset, please refer to [Section 2.3: Data Organization for Building a Feature Library](#23-data-organization-for-building-a-feature-library). Afterward, you can complete the establishment of the facial feature library and quickly perform inference with the facial recognition pipeline using just a few lines of code.

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

index_data = pipeline.build_index(gallery_imgs="face_demo_gallery", gallery_label="face_demo_gallery/gallery.txt")
index_data.save("face_index")

output = pipeline.predict("friends1.jpg", index=index_data)
for res in output:
    res.print()
    res.save_to_img("./output/")
```

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a face recognition pipeline object. The specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the pipeline name, it must be a pipeline supported by PaddleX.</td>
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
(2) Call the `build_index` method of the face recognition pipeline object to build the facial feature library. The specific parameters are described as follows:

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
<td><code>gallery_imgs</code></td>
<td>Base library images to be added, supported formats: 1. <code>str</code> type representing the root directory of images, with data organization consistent with the method used when constructing the index library, refer to <a href="#2.3-Data-Organization-for-Building-a-Feature-Library">Section 2.3: Data Organization for Building a Feature Library</a>; 2. <code>[numpy.ndarray, numpy.ndarray, ..]</code> type base library image data.</td>
<td><code>str</code>|<code>list</code></td>
<td>None</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>Annotation information for base library images, supported formats: 1. <code>str</code> type representing the path to the annotation file, with data organization consistent with the method used when constructing the feature library, refer to <a href="#2.3-Data-Organization-for-Building-a-Feature-Library">Section 2.3: Data Organization for Building a Feature Library</a>; 2. <code>[str, str, ..]</code> type representing the annotations of base library images.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>

The feature library object `index` supports the `save` method, which is used to save the feature library to disk:

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
<td>The directory to save the feature library file, e.g., <code>drink_index</code>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>

(3) Call the `predict` method of the face recognition pipeline object for inference prediction: The `predict` method parameter is `x`, used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

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
<td>Supports directly passing in Python variables, such as image data represented by <code>numpy.ndarray</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the file path of the data to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/friends1.jpg">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in a local directory containing the data files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing in a dictionary type, where the key needs to correspond to the specific task, such as "img" for image classification tasks, and the value of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing in a list, where the list elements need to be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
(4) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

(5) Process the prediction results: The prediction result for each sample is of type `dict`, and it supports printing or saving to a file. The supported file types depend on the specific pipeline, such as:

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
<td>Print results to the terminal</td>
<td><code>- format_json</code>: Boolean, whether to format the output with JSON indentation, default is True; <br/><code>- indent</code>: Integer, JSON formatting setting, effective only when format_json is True, default is 4; <br/><code>- ensure_ascii</code>: Boolean, JSON formatting setting, effective only when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Save results as a JSON file</td>
<td><code>- save_path</code>: String, file path for saving; if it's a directory, the saved file name matches the input file name; <br/><code>- indent</code>: Integer, JSON formatting setting, default is 4; <br/><code>- ensure_ascii</code>: Boolean, JSON formatting setting, default is False;</td>
</tr>
<tr>
<td>save_to_img</td>
<td>Save results as an image file</td>
<td><code>- save_path</code>: String, file path for saving; if it's a directory, the saved file name matches the input file name;</td>
</tr>
</tbody>
</table>
If you have obtained the configuration file, you can customize various settings of the facial recognition pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/face_recognition.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/face_recognition.yaml", index="face_index")

output = pipeline.predict("friends1.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

#### 2.2.3 Adding and Deleting Operations in the Face Feature Library

If you wish to add more face images to the feature library, you can call the `append_index` method; to delete face image features, you can call the `remove_index` method.

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

index_data = pipeline.build_index(gallery_imgs="face_demo_gallery", gallery_label="face_demo_gallery/gallery.txt", index_type="IVF", metric_type="IP")
index_data = pipeline.append_index(gallery_imgs="face_demo_gallery", gallery_label="face_demo_gallery/gallery.txt", index=index_data)
index_data = pipeline.remove_index(remove_ids="face_demo_gallery/remove_ids.txt", index=index_data)
index_data.save("face_index")
```

The `add_index` method parameters are described as follows:

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
<td><code>gallery_imgs</code></td>
<td>Base library images to be added, supported formats: 1. <code>str</code> type representing the root directory of images, with data organization consistent with the method used when constructing the index library, refer to <a href="#2.3-Data-Organization-for-Building-a-Feature-Library">Section 2.3: Data Organization for Building a Feature Library</a>; 2. <code>[numpy.ndarray, numpy.ndarray, ..]</code> type base library image data.</td>
<td><code>str</code>|<code>list</code></td>
<td>None</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>Annotation information for base library images, supported formats: 1. <code>str</code> type representing the path to the annotation file, with data organization consistent with the method used when constructing the feature library, refer to <a href="#2.3-Data-Organization-for-Building-a-Feature-Library">Section 2.3: Data Organization for Building a Feature Library</a>; 2. <code>[str, str, ..]</code> type representing the annotations of base library images.</td>
<td><code>str</code>|<code>list</code></td>
<td>None</td>
</tr>
<tr>
<td><code>remove_ids</code></td>
<td>Index numbers to be removed, supported formats: 1. <code>str</code> type representing the path of a txt file, with content being the IDs of indexes to be removed, one "id" per line; 2. <code>[int, int, ..]</code> type representing the index numbers to be removed. Only effective in <code>remove_index</code>.</td>
<td><code>str</code>|<code>list</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index</code></td>
<td>Feature library, supported formats: 1. The path to the directory containing the feature library files (<code>vector.index</code> and <code>index_info.yaml</code>); 2. <code>IndexData</code> type feature library object, only effective in <code>append_index</code> and <code>remove_index</code>, representing the feature library to be modified.</td>
<td><code>str</code>|<code>IndexData</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>Supports <code>HNSW32</code>, <code>IVF</code>, <code>Flat</code>. Among them, <code>HNSW32</code> offers fast retrieval speed and high accuracy, but does not support the <code>remove_index()</code> operation; <code>IVF</code> offers fast retrieval speed but relatively lower accuracy, supporting both <code>append_index()</code> and <code>remove_index()</code> operations; <code>Flat</code> offers lower retrieval speed but higher accuracy, supporting both <code>append_index()</code> and <code>remove_index()</code> operations.</td>
<td><code>str</code></td>
<td><code>HNSW32</code></td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>Supports: <code>IP</code>, Inner Product; <code>L2</code>, Euclidean Distance.</td>
<td><code>str</code></td>
<td><code>IP</code></td>
</tr>
</tbody>
</table>

### 2.3 Data Organization for Feature Library Construction

The face recognition pipeline example in PaddleX requires a pre-constructed feature library for face feature retrieval. If you wish to build a face feature library with private data, you need to organize the data as follows:

```bash
data_root             # Root directory of the dataset, the directory name can be changed
├── images            # Directory for saving images, the directory name can be changed
│   ├── ID0           # Identity ID name, preferably meaningful, such as a person's name
│   │   ├── xxx.jpg   # Image, nested directories are supported
│   │   ├── xxx.jpg   # Image, nested directories are supported
│   │       ...
│   ├── ID1           # Identity ID name, preferably meaningful, such as a person's name
│   │   ├── xxx.jpg   # Image, nested directories are supported
│   │   ├── xxx.jpg   # Image, nested directories are supported
│   │       ...
│       ...
└── gallery.txt       # Annotation file for the feature library dataset, the file name cannot be changed. Each line gives the path of the face image to be retrieved and the image feature label, separated by a space. Example content: images/Chandler/Chandler00037.jpg Chandler
```

## 3. Development Integration/Deployment
If the face recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to directly apply the face recognition pipeline in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

<details><summary>API Reference</summary>

<p>For primary operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and the response body are both JSON data (JSON objects).</li>
<li>When the request is successfully processed, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
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
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed to <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed to <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not successfully processed, the attributes of the response body are as follows:</li>
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
<p>The primary operations provided by the service are as follows:</p>
<ul>
<li><b><code>buildIndex</code></b></li>
</ul>
<p>Build feature vector index.</p>
<p><code>POST /face-recognition-index-build</code></p>
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
<td>The URL of the image file accessible to the server or the Base64 encoded content of the image file.</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>Label.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> in the response body has the following properties:</li>
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
<td>The key corresponding to the index, used to identify the established index. Can be used as input for other operations.</td>
</tr>
<tr>
<td><code>idMap</code></td>
<td><code>object</code></td>
<td>Mapping from vector ID to label.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>addImagesToIndex</code></b></li>
</ul>
<p>Add images (corresponding feature vectors) to the index.</p>
<p><code>POST /face-recognition-index-add</code></p>
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
<td>The URL of the image file accessible to the server or the Base64 encoded content of the image file.</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>Label.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> in the response body has the following properties:</li>
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
<td>Mapping from vector ID to label.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>removeImagesFromIndex</code></b></li>
</ul>
<p>Remove images (corresponding feature vectors) from the index.</p>
<p><code>POST /face-recognition-index-remove</code></p>
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
<td>IDs of vectors to be removed from the index.</td>
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
<li>When the request is successfully processed, the <code>result</code> in the response body has the following properties:</li>
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
<td>Mapping from vector ID to label.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform image recognition.</p>
<p><code>POST /face-recognition-infer</code></p>
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
<td>The URL of the image file accessible to the server or the Base64 encoded content of the image file.</td>
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
<li>When the request is successfully processed, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>faces</code></td>
<td><code>array</code></td>
<td>Information about detected faces.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>Recognition result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>faces</code> is an <code>object</code> with the following properties:</p>
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
<td>Face target position. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner, in order.</td>
</tr>
<tr>
<td><code>recResults</code></td>
<td><code>array</code></td>
<td>Recognition results.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Detection score.</td>
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
<td>Label.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Recognition score.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<pre><code class="language-python">import base64
import pprint
import sys

import requests

API_BASE_URL = &quot;http://0.0.0.0:8080&quot;

base_image_label_pairs = [
    {&quot;image&quot;: &quot;./demo0.jpg&quot;, &quot;label&quot;: &quot;ID0&quot;},
    {&quot;image&quot;: &quot;./demo1.jpg&quot;, &quot;label&quot;: &quot;ID1&quot;},
    {&quot;image&quot;: &quot;./demo2.jpg&quot;, &quot;label&quot;: &quot;ID2&quot;},
]
image_label_pairs_to_add = [
    {&quot;image&quot;: &quot;./demo3.jpg&quot;, &quot;label&quot;: &quot;ID2&quot;},
]
ids_to_remove = [1]
infer_image_path = &quot;./demo4.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

for pair in base_image_label_pairs:
    with open(pair[&quot;image&quot;], &quot;rb&quot;) as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)
    pair[&quot;image&quot;] = image_data

payload = {&quot;imageLabelPairs&quot;: base_image_label_pairs}
resp_index_build = requests.post(f&quot;{API_BASE_URL}/face-recognition-index-build&quot;, json=payload)
if resp_index_build.status_code != 200:
    print(f&quot;Request to face-recognition-index-build failed with status code {resp_index_build}.&quot;)
    pprint.pp(resp_index_build.json())
    sys.exit(1)
result_index_build = resp_index_build.json()[&quot;result&quot;]
print(f&quot;Number of images indexed: {len(result_index_build['idMap'])}&quot;)

for pair in image_label_pairs_to_add:
    with open(pair[&quot;image&quot;], &quot;rb&quot;) as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)
    pair[&quot;image&quot;] = image_data

payload = {&quot;imageLabelPairs&quot;: image_label_pairs_to_add, &quot;indexKey&quot;: result_index_build[&quot;indexKey&quot;]}
resp_index_add = requests.post(f&quot;{API_BASE_URL}/face-recognition-index-add&quot;, json=payload)
if resp_index_add.status_code != 200:
    print(f&quot;Request to face-recognition-index-add failed with status code {resp_index_add}.&quot;)
    pprint.pp(resp_index_add.json())
    sys.exit(1)
result_index_add = resp_index_add.json()[&quot;result&quot;]
print(f&quot;Number of images indexed: {len(result_index_add['idMap'])}&quot;)

payload = {&quot;ids&quot;: ids_to_remove, &quot;indexKey&quot;: result_index_build[&quot;indexKey&quot;]}
resp_index_remove = requests.post(f&quot;{API_BASE_URL}/face-recognition-index-remove&quot;, json=payload)
if resp_index_remove.status_code != 200:
    print(f&quot;Request to face-recognition-index-remove failed with status code {resp_index_remove}.&quot;)
    pprint.pp(resp_index_remove.json())
    sys.exit(1)
result_index_remove = resp_index_remove.json()[&quot;result&quot;]
print(f&quot;Number of images indexed: {len(result_index_remove['idMap'])}&quot;)

with open(infer_image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data, &quot;indexKey&quot;: result_index_build[&quot;indexKey&quot;]}
resp_infer = requests.post(f&quot;{API_BASE_URL}/face-recognition-infer&quot;, json=payload)
if resp_infer.status_code != 200:
    print(f&quot;Request to face-recogntion-infer failed with status code {resp_infer}.&quot;)
    pprint.pp(resp_infer.json())
    sys.exit(1)
result_infer = resp_infer.json()[&quot;result&quot;]

with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result_infer[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nDetected faces:&quot;)
pprint.pp(result_infer[&quot;faces&quot;])
</code></pre>
</details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method where computing and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose an appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.


## 4. Custom Development
If the default model weights provided by the Face Recognition Pipeline do not meet your expectations in terms of accuracy or speed for your specific scenario, you can try to further <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to enhance the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the Face Recognition Pipeline consists of two modules (face detection and face recognition), the suboptimal performance of the pipeline may stem from either module.

You can analyze images with poor recognition results. If you find that many faces are not detected during the analysis, it may indicate deficiencies in the face detection model. In this case, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/face_detection.en.md#IV.-Custom-Development) section in the [Face Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_detection.en.md) and use your private dataset to fine-tune the face detection model. If matching errors occur in detected faces, it suggests that the face feature model needs further improvement. You should refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/face_feature.en.md#IV.-Custom-Development) section in the [Face Feature Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_feature.en.md) to fine-tune the face feature model.

### 4.2 Model Application
After completing fine-tuning training with your private dataset, you will obtain local model weight files.

To use the fine-tuned model weights, you only need to modify the pipeline configuration file by replacing the local paths of the fine-tuned model weights with the corresponding paths in the pipeline configuration file:

```bash

......
Pipeline:
  device: "gpu:0"
  det_model: "BlazeFace"        # Can be modified to the local path of the fine-tuned face detection model
  rec_model: "MobileFaceNet"    # Can be modified to the local path of the fine-tuned face recognition model
  det_batch_size: 1
  rec_batch_size: 1
  device: gpu
......
```
Subsequently, refer to the command-line method or Python script method in [2.2 Local Experience](#22-Local-Experience) to load the modified pipeline configuration file.
Note: Currently, setting separate `batch_size` for face detection and face recognition models is not supported.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modifying the `--device` parameter</b> allows seamless switching between different hardware.

For example, when running the face recognition pipeline using Python and changing the running device from an NVIDIA GPU to an Ascend NPU, you only need to modify the `device` in the script to `npu`:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="face_recognition",
    device="npu:0" # gpu:0 --> npu:0
)
```
If you want to use the face recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
