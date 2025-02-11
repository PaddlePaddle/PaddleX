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
<th>CPU Inference Time (ms)</th>
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
> ❗ Before using the face recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience

Command line experience is not supported yet.

#### 2.2.2 Python Script Integration
Please download the [test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/friends1.jpg) for testing.</url>
In the example run of this pipeline, you need to pre-build a face feature library. You can refer to the following instructions to download the official demo data for subsequent construction of the face feature library.
You can refer to the following command to download the Demo dataset to the specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_demo_gallery.tar
tar -xf ./face_demo_gallery.tar
```

If you wish to build a facial feature library using your private dataset, you can refer to [Section 2.3 Data Organization for Building Feature Libraries](). After that, you can complete the establishment of the facial feature library and the fast inference of the facial recognition production line with just a few lines of code.

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

index_data = pipeline.build_index(gallery_imgs="face_demo_gallery", gallery_label="face_demo_gallery/gallery.txt")
index_data.save("face_index")

output = pipeline.predict("friends1.jpg", index=index_data)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

In the above Python script, the following steps are performed:

(1) Call `create_pipeline` to instantiate the face recognition production line object. The specific parameter descriptions are as follows:

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
<td>The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be a production line supported by PaddleX.</td>
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
<td>The inference device for the production line. Supports specifying the specific card number of the GPU, such as "gpu:0", the specific card number of other hardware, such as "npu:0", and CPU such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, available only when the production line supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `build_index` method of the face recognition production line object to build the face feature library. The specific parameter descriptions are as follows:

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
<td>The base library images to be added, required parameter</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
  <li><b>str</b>: The root directory of the images, data organization method refers to <a href="#2.3-构建特征库的数据组织方式">Section 2.3 Data Organization Method for Building Feature Library</a></li>
  <li><b>List[numpy.ndarray]</b>: List of numpy.array type base library image data</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>The annotation information of the base library images, required parameter</td>
<td><code>str|list</code></td>
<td>
<ul>
  <li><b>str</b>: The path to the annotation file, the data organization method is the same as when building the feature library, refer to <a href="#2.3-构建特征库的数据组织方式">Section 2.3 Data Organization Method for Building Feature Library</a></li>
  <li><b>List[str]</b>: List of str type base library image annotations</li>
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
<td>Index type, optional parameter</td>
<td><code>str</code></td>
<td>
<ul>
  <li><code>"HNSW32"</code>: Fast retrieval speed and high accuracy, but does not support <code>remove_index()</code> operation</li>
  <li><code>"IVF"</code>: Fast retrieval speed but relatively low accuracy, supports <code>append_index()</code> and <code>remove_index()</code> operations</li>
  <li><code>"Flat"</code>: Low retrieval speed and high accuracy, supports <code>append_index()</code> and <code>remove_index()</code> operations</li>
</ul>
</td>
<td><code>"HNSW32"</code></td>
</tr>
</tbody>
</table>

- The feature library object `index` supports the `save` method to save the feature library to disk:

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
<td>The save directory of the feature library file, such as <code>drink_index</code>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>

(3) Call the `predict` method of the face recognition pipeline object for inference prediction: The parameter of the `predict` method is `input`, which is used to input the data to be predicted and supports multiple input methods. Specific examples are as follows:

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
<td>Data to be predicted, supports multiple input types (required parameter)</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
  <li><b>str</b>: Local path of an image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as a network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code></li>
  <li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>index</code></td>
<td>The feature library used for pipeline inference prediction (optional parameter). If this parameter is not provided, the index library specified in the pipeline configuration file will be used by default.</td>
<td><code>str|paddlex.inference.components.retrieval.faiss.IndexData|None</code></td>
<td>
<ul>
    <li><b>str</b> type representing a directory (which should contain feature library files, including <code>vector.index</code> and <code>index_info.yaml</code>)</li>
    <li><b>IndexData</b> object created by the <code>build_index</code> method</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

(4) Process the prediction results: The prediction result of each sample is of `dict` type and supports printing or saving to a file. The supported file types depend on the specific pipeline, such as:

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
<td>Path to save the file. When it is a directory, the saved file name will be consistent with the input file type</td>
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
{'res': {'input_path': 'friends1.jpg', 'boxes': [{'labels': ['Chandler', 'Chandler', 'Chandler', 'Chandler', 'Chandler'], 'rec_scores': [0.5884832143783569, 0.5777347087860107, 0.5082703828811646, 0.48792028427124023, 0.4842316806316376], 'det_score': 0.9119220972061157, 'coordinate': [790.40015, 170.34453, 868.47626, 279.54446]}, {'labels': ['Joey', 'Joey', 'Joey', 'Joey', 'Joey'], 'rec_scores': [0.5654032826423645, 0.5601680278778076, 0.5382657051086426, 0.5320160984992981, 0.5209866762161255], 'det_score': 0.9052104353904724, 'coordinate': [1274.6246, 184.58124, 1353.4016, 300.0643]}, {'labels': ['Phoebe', 'Phoebe', 'Phoebe', 'Phoebe', 'Phoebe'], 'rec_scores': [0.6462339162826538, 0.6003466844558716, 0.5999515652656555, 0.583031415939331, 0.5640993118286133], 'det_score': 0.9041699171066284, 'coordinate': [1052.4514, 192.52296, 1129.5226, 292.84177]}, {'labels': ['Ross', 'Ross', 'Ross', 'Ross', 'Ross'], 'rec_scores': [0.5012176036834717, 0.49081552028656006, 0.48970693349838257, 0.4808862805366516, 0.4794950783252716], 'det_score': 0.9031845331192017, 'coordinate': [162.41049, 156.96768, 242.07184, 266.13004]}, {'labels': ['Monica', 'Monica', 'Monica', 'Monica', 'Monica'], 'rec_scores': [0.5704089403152466, 0.5037636756896973, 0.4877302646636963, 0.46702104806900024, 0.4376206696033478], 'det_score': 0.8862134218215942, 'coordinate': [572.18176, 216.25815, 639.2387, 311.08417]}, {'labels': ['Rachel', 'Rachel', 'Rachel', 'Rachel', 'Rachel'], 'rec_scores': [0.6107711791992188, 0.5915063619613647, 0.5776835083961487, 0.569993257522583, 0.5594189167022705], 'det_score': 0.8822972774505615, 'coordinate': [303.12866, 231.94759, 374.5314, 330.2883]}]}}
```

- The meanings of the output parameters are as follows:
    - `input_path`: Indicates the path of the input image.
    - `boxes`: Information of detected faces, a list of dictionaries, each dictionary contains the following information:
        - `labels`: List of recognized labels, sorted by score from high to low.
        - `rec_scores`: List of recognition scores, where elements correspond to `labels` one by one.
        - `det_score`: Detection score.
        - `coordinate`: Coordinates of the face bounding box, in the format [xmin, ymin, xmax, ymax].

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file.
- Calling the `save_to_img()` method will save the visualization result to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The production line usually contains many result images; it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, leaving only the last one.) In the example above, the visualization result is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/face_recognition/02.jpg">

* Additionally, it also supports obtaining the visualized image with results and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction result in <code>json</code> format.</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">Get the visualized image in <code>dict</code> format.</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is data of dict type, and the relevant content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of dict type. The key is `res`, and the corresponding value is an `Image.Image` object used to visualize the face recognition result.

The above Python script integration method uses the parameter settings in the PaddleX official configuration file by default. If you need to customize the configuration file, you can first execute the following command to obtain the official configuration file and save it in `my_path`:

```bash
paddlex --get_pipeline_config face_recognition --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the face recognition production line. You just need to modify the `pipeline` parameter value in the `create_pipeline` method to the path of your custom production line configuration file.

For example, if your custom configuration file is saved in `./my_path/face_recognition.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/face_recognition.yaml")

output = pipeline.predict("friends1.jpg", index="face_index")
for res in output:
    res.print()
    res.save_to_json("./output/")
    res.save_to_img("./output/")
```

<b>Note:</b> The parameters in the configuration file are the initialization parameters of the pipeline. If you wish to change the initialization parameters of the face recognition pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction.

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
  <li><b>str</b>: Root directory of images, data organization refers to <a href="#2.3-Data Organization for Building the Feature Library">Section 2.3 Data Organization for Building the Feature Library</a></li>
  <li><b>List[numpy.ndarray]</b>: Gallery image data in the form of a list of numpy arrays</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>Labels for gallery images, required parameter</td>
<td><code>str|list</code></td>
<td>
<ul>
  <li><b>str</b>: Path to the label file, data organization is the same as when building the feature library, refer to <a href="#2.3-Data Organization for Building the Feature Library">Section 2.3 Data Organization for Building the Feature Library</a></li>
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

### 2.3 Data Organization for Building the Feature Library
The face recognition pipeline example of PaddleX requires a pre-built feature library for face feature retrieval. If you wish to build a face feature library with your private data, you need to organize the data as follows:

```bash
data_root             # Root directory of the dataset, the directory name can be changed
├── images            # Directory for storing images, the directory name can be changed
│   ├── ID0           # Identity ID name, preferably a meaningful name, such as a person's name
│   │   ├── xxx.jpg   # Image, nested levels are supported here
│   │   ├── xxx.jpg   # Image, nested levels are supported here
│   │       ...
│   ├── ID1           # Identity ID name, preferably a meaningful name, such as a person's name
│   │   ├── xxx.jpg   # Image, nested levels are supported here
│   │   ├── xxx.jpg   # Image, nested levels are supported here
│   │       ...
│       ...
└── gallery.txt       # Annotation file for the feature library dataset, the file name can be changed. Each line provides the path and label of the face image to be retrieved, separated by a space. Example content: images/Chandler/Chandler00037.jpg Chandler
```

## 3. Development Integration/Deployment
If the face recognition production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the face recognition production line directly in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

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
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
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
<td>The URL of the image file accessible by the server or the Base64-encoded result of the image file content.</td>
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
<th>Meaning</th>
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
<td><code>faces</code></td>
<td><code>array</code></td>
<td>Information about detected faces.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The recognition result image. The image is in JPEG format and is Base64-encoded.</td>
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
<td>The location of the face target. The elements of the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.</td>
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

API_BASE_URL = &quot;<url id="cu9nt776o68pmutlr330" type="url" status="failed" title="" wc="0">http://0.0.0.0:8080&quot;</url>

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

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate method to deploy the model pipeline according to your needs, and then proceed with subsequent AI application integration.


## 4. Secondary Development
If the default model weights provided by the face recognition pipeline do not meet your accuracy or speed requirements in your scenario, you can try further <b>fine-tuning</b> the existing model using <b>your own specific domain or application data</b> to improve the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the face recognition pipeline includes two modules (face detection and face feature), the unsatisfactory performance of the model pipeline may come from either module.

You can analyze the images with poor recognition performance. If you find that many faces are not detected during the analysis, it may indicate a deficiency in the face detection model. You need to refer to the [Face Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_detection.en.md) and the [Secondary Development](../../../module_usage/tutorials/cv_modules/face_detection.en.md) section to fine-tune the face detection model using your private dataset. If there are matching errors in the detected faces, it indicates that the face feature module needs further improvement. You need to refer to the [Face Feature Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_feature.md) and the [Secondary Development](../../../module_usage/tutorials/cv_modules/face_feature.md) section to fine-tune the face feature module.

### 4.2 Model Application
After completing the fine-tuning training with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, you only need to modify the pipeline configuration file by replacing the local path of the fine-tuned model weights in the corresponding position of the pipeline configuration file:

```yaml

...

SubModules:
  Detection:
    module_name: face_detection
    model_name: PP-YOLOE_plus-S_face
    model_dir: null #可修改为微调后人脸检测模型的本地路径
    batch_size: 1
  Recognition:
    module_name: face_feature
    model_name: ResNet50_face
    model_dir: null #可修改为微调后人脸特征模型的本地路径
    batch_size: 1
```

Subsequently, refer to the command-line or Python script methods in [2.2 Local Experience]() to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, when running the face recognition production line using Python, to change the runtime device from NVIDIA GPU to Ascend NPU, just modify the `device` in the script to `npu`:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="face_recognition",
    device="npu:0" # gpu:0 --> npu:0
    )
```

If you want to use the face recognition production line on more types of hardware, please refer to [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
