---
comments: true
---

# General Instance Segmentation Pipeline Tutorial

## 1. Introduction to the General Instance Segmentation Pipeline
Instance segmentation is a computer vision task that not only identifies the object categories in an image but also distinguishes the pixels of different instances within the same category, enabling precise segmentation of each object. Instance segmentation can separately label each car, person, or animal in an image, ensuring they are independently processed at the pixel level. For example, in a street scene image containing multiple cars and pedestrians, instance segmentation can clearly separate the contours of each car and person, forming multiple independent region labels. This technology is widely used in autonomous driving, video surveillance, and robotic vision, often relying on deep learning models (such as Mask R-CNN) to achieve efficient pixel classification and instance differentiation through Convolutional Neural Networks (CNNs), providing powerful support for understanding complex scenes.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/01.png">


<b>The General Instance Segmentation Pipeline includes a</b> <b>Object Detection</b> <b>module. If you prioritize model precision, choose a model with higher precision. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Mask AP</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>Mask-RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>50.6</td>
<td>132.693</td>
<td>4896.17</td>
<td>449.9 M</td>
<td rowspan="5">Mask-RT-DETR is an instance segmentation model based on RT-DETR. By adopting the high-performance PP-HGNetV2 as the backbone network and constructing a MaskHybridEncoder encoder, along with introducing IOU-aware Query Selection technology, it achieves state-of-the-art (SOTA) instance segmentation accuracy with the same inference time.</td>
</tr>
<tr>
<td>Mask-RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>45.7</td>
<td>46.5059</td>
<td>2575.92</td>
<td>113.6 M</td>
</tr>
</table>

> ❗ The above list features the <b>2 core models</b> that the image classification module primarily supports. In total, this module supports <b>15 models</b>. The complete list of models is as follows:

<details><summary> 👉Model List Details</summary>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Mask AP</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>36.3</td>
<td>-</td>
<td>-</td>
<td>254.8 M</td>
<td rowspan="2">Cascade-MaskRCNN is an improved Mask RCNN instance segmentation model that utilizes multiple detectors in a cascade, optimizing segmentation results by leveraging different IOU thresholds to address the mismatch between detection and inference stages, thereby enhancing instance segmentation accuracy.</td>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Trained Model</a></td>
<td>39.1</td>
<td>-</td>
<td>-</td>
<td>254.7 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>50.6</td>
<td>132.693</td>
<td>4896.17</td>
<td>449.9 M</td>
<td rowspan="5">Mask-RT-DETR is an instance segmentation model based on RT-DETR. By adopting the high-performance PP-HGNetV2 as the backbone network and constructing a MaskHybridEncoder encoder, along with introducing IOU-aware Query Selection technology, it achieves state-of-the-art (SOTA) instance segmentation accuracy with the same inference time.</td>
</tr>
<tr>
<td>Mask-RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>45.7</td>
<td>46.5059</td>
<td>2575.92</td>
<td>113.6 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-M_pretrained.pdparams">Trained Model</a></td>
<td>42.7</td>
<td>36.8329</td>
<td>-</td>
<td>66.6 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-S_pretrained.pdparams">Trained Model</a></td>
<td>41.0</td>
<td>33.5007</td>
<td>-</td>
<td>51.8 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-X</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-X_pretrained.pdparams">Trained Model</a></td>
<td>47.5</td>
<td>75.755</td>
<td>3358.04</td>
<td>237.5 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>35.6</td>
<td>-</td>
<td>-</td>
<td>157.5 M</td>
<td rowspan="6">Mask R-CNN is a full-task deep learning model from Facebook AI Research (FAIR) that can perform object classification and localization in a single model, combined with image-level masks to complete segmentation tasks.</td>
</tr>
<tr>
<td>MaskRCNN-ResNet50-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>36.4</td>
<td>-</td>
<td>-</td>
<td>157.5 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>32.8</td>
<td>-</td>
<td>-</td>
<td>128.7 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet101-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-FPN_pretrained.pdparams">Trained Model</a></td>
<td>36.6</td>
<td>-</td>
<td>-</td>
<td>225.4 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet101-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>38.1</td>
<td>-</td>
<td>-</td>
<td>225.1 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNeXt101-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNeXt101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>39.5</td>
<td>-</td>
<td>-</td>
<td>370.0 M</td>
<td></td>
</tr>
<tr>
<td>PP-YOLOE_seg-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_seg-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_seg-S_pretrained.pdparams">Trained Model</a></td>
<td>32.5</td>
<td>-</td>
<td>-</td>
<td>31.5 M</td>
<td>PP-YOLOE_seg is an instance segmentation model based on PP-YOLOE. This model inherits PP-YOLOE's backbone and head, significantly enhancing instance segmentation performance and inference speed through the design of a PP-YOLOE instance segmentation head.</td>
</tr>
<tr>
<td>SOLOv2</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SOLOv2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SOLOv2_pretrained.pdparams">Trained Model</a></td>
<td>35.5</td>
<td>-</td>
<td>-</td>
<td>179.1 M</td>
<td> SOLOv2 is a real-time instance segmentation algorithm that segments objects by location. This model is an improved version of SOLO, achieving a good balance between accuracy and speed through the introduction of mask learning and mask NMS.</td>
</tr>
</table>

<p><b>Note: The above accuracy metrics are based on the Mask AP of the <a href="https://cocodataset.org/#home">COCO2017</a> validation set. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p></details>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX allow for quick experience of the effects. You can experience the effects of the General Instance Segmentation Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/100063/webUI) the effects of the General Instance Segmentation Pipeline using the demo images provided by the official. For example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/02.png">

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the model within the pipeline</b>.

### 2.2 Local Experience
Before using the General Image Classification Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
A single command is all you need to quickly experience the image classification pipeline, Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png), and replace `--input` with the local path to perform prediction.


```bash
paddlex --pipeline instance_segmentation --input  general_instance_segmentation_004.png --device gpu:0
```

Parameter Description:

```
--pipeline: The name of the pipeline, here it refers to the object detection pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 indicates using the first GPU, gpu:1,2 indicates using the second and third GPUs), or you can choose to use CPU (--device cpu).
```

When executing the above Python script, the default instance segmentation pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> 👉Click to expand</summary>

<pre><code>paddlex --get_pipeline_config instance_segmentation
</code></pre>
<p>After execution, the instance segmentation pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code>paddlex --get_pipeline_config instance_segmentation --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, you can replace <code>--pipeline</code> with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is <code>./instance_segmentation.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./instance_segmentation.yaml --input general_instance_segmentation_004.png --device gpu:0
</code></pre>
<p>Where <code>--model</code>, <code>--device</code>, and other parameters do not need to be specified, and the parameters in the configuration file will be used. If parameters are still specified, the specified parameters will take precedence.</p></details>

After running, the result is:

```
{'input_path': 'general_instance_segmentation_004.png', 'boxes': [{'cls_id': 0, 'label': 'person', 'score': 0.8698326945304871, 'coordinate': [339, 0, 639, 575]}, {'cls_id': 0, 'label': 'person', 'score': 0.8571141362190247, 'coordinate': [0, 0, 195, 575]}, {'cls_id': 0, 'label': 'person', 'score': 0.8202633857727051, 'coordinate': [88, 113, 401, 574]}, {'cls_id': 0, 'label': 'person', 'score': 0.7108577489852905, 'coordinate': [522, 21, 767, 574]}, {'cls_id': 27, 'label': 'tie', 'score': 0.554280698299408, 'coordinate': [247, 311, 355, 574]}]}
```

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/03.png">

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

#### 2.2.2 Python Script Integration
A few lines of code can complete the quick inference of the pipeline. Taking the general instance segmentation pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="instance_segmentation")

output = pipeline.predict("general_instance_segmentation_004.png")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the visualization image of the result
    res.save_to_json("./output/") # Save the structured output of the prediction
```
The results obtained are the same as those obtained through the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: The specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
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
<td>Whether to enable high-performance inference, which is only available when the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `predict` method of the image classification pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

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
<td>Supports directly passing Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing a dictionary type, where the key needs to correspond to the specific task, such as "img" for the image classification task, and the value of the dictionary supports the above data types, e.g., <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing a list, where the list elements need to be the above data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
3）Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

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
If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/instance_segmentation.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/instance_segmentation.yaml")
output = pipeline.predict("general_instance_segmentation_004.png")
for res in output:
    res.print() # Print the structured output of prediction
    res.save_to_img("./output/") # Save the visualized image of the result
    res.save_to_json("./output/") # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing for significant speedups in the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

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
<td>Error message. Fixed as <code>"Success"</code>.</td>
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
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>Primary operations provided by the service:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Performs instance segmentation on an image.</p>
<p><code>POST /instance-segmentation</code></p>
<ul>
<li>The request body properties are as follows:</li>
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
<td>The URL of an image file accessible by the server or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following properties:</li>
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
<td><code>instances</code></td>
<td><code>array</code></td>
<td>Information about the locations and categories of instances.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The result image of instance segmentation. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>instances</code> is an <code>object</code> with the following properties:</p>
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
<td>The location of the instance. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively.</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>The ID of the instance category.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The score of the instance.</td>
</tr>
<tr>
<td><code>mask</code></td>
<td><code>object</code></td>
<td>The segmentation mask of the instance.</td>
</tr>
</tbody>
</table>
<p>The properties of <code>mask</code> are as follows:</p>
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
<td><code>rleResult</code></td>
<td><code>str</code></td>
<td>The run-length encoding result of the mask.</td>
</tr>
<tr>
<td><code>size</code></td>
<td><code>array</code></td>
<td>The shape of the mask. The elements in the array are the height and width of the mask, respectively.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;instances&quot;: [
{
&quot;bbox&quot;: [
162.39381408691406,
83.88176727294922,
624.0797119140625,
343.4986877441406
],
&quot;categoryId&quot;: 33,
&quot;score&quot;: 0.8691174983978271,
&quot;mask&quot;: {
&quot;rleResult&quot;: &quot;xxxxxx&quot;,
&quot;size&quot;: [
259,
462
]
}
}
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/instance-segmentation&quot;
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nInstances:&quot;)
print(result[&quot;instances&quot;])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

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

    auto response = client.Post(&quot;/instance-segmentation&quot;, headers, body, &quot;application/json&quot;);
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

        auto instances = result[&quot;instances&quot;];
        std::cout &lt;&lt; &quot;\nInstances:&quot; &lt;&lt; std::endl;
        for (const auto&amp; inst : instances) {
            std::cout &lt;&lt; inst &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/instance-segmentation&quot;;
        String imagePath = &quot;./demo.jpg&quot;;
        String outputImagePath = &quot;./out.jpg&quot;;

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode instances = result.get(&quot;instances&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
                System.out.println(&quot;\nInstances: &quot; + instances.toString());
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/instance-segmentation&quot;
    imagePath := &quot;./demo.jpg&quot;
    outputImagePath := &quot;./out.jpg&quot;

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:&quot;image&quot;`
            Instances []map[string]interface{} `json:&quot;instances&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, outputImagePath)
    fmt.Println(&quot;\nInstances:&quot;)
    for _, inst := range respData.Result.Instances {
        fmt.Println(inst)
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
    static readonly string API_URL = &quot;http://localhost:8080/instance-segmentation&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nInstances:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;instances&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/instance-segmentation'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)
  })
};

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log(&quot;\nInstances:&quot;);
    console.log(result[&quot;instances&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/instance-segmentation&quot;;
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nInstances:\n&quot;;
print_r($result[&quot;instances&quot;]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing functions on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.

## 4. CCustom Development
If the default model weights provided by the general instance segmentation pipeline do not meet your requirements for accuracy or speed in your scenario, you can try to further <b>fine-tune</b> the existing model using <b>data specific to your domain or application scenario</b> to improve the recognition effect of the general instance segmentation pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general instance segmentation pipeline includes an instance segmentation module, if the performance of the pipeline does not meet expectations, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/instance_segmentation.en.md#iv-custom-development) section in the [Instance Segmentation Module Development Tutorial](../../../module_usage/tutorials/cv_modules/instance_segmentation.en.md).

### 4.2 Model Application
After you complete fine-tuning training using your private dataset, you will obtain local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: Mask-RT-DETR-S  # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```

Then, refer to the command line method or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for instance segmentation pipeline inference, the Python command is:

```bash
paddlex --pipeline instance_segmentation --input general_instance_segmentation_004.png --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline instance_segmentation --input general_instance_segmentation_004.png --device npu:0
```

If you want to use the General Instance Segmentation Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
