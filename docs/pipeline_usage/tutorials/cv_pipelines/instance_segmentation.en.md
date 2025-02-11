---
comments: true
---

# General Instance Segmentation Pipeline Tutorial

## 1. Introduction to the General Instance Segmentation Pipeline
Instance segmentation is a computer vision task that not only identifies the object categories in an image but also distinguishes the pixels of different instances within the same category, enabling precise segmentation of each object. Instance segmentation can separately label each car, person, or animal in an image, ensuring they are independently processed at the pixel level. For example, in a street scene image containing multiple cars and pedestrians, instance segmentation can clearly separate the contours of each car and person, forming multiple independent region labels. This technology is widely used in autonomous driving, video surveillance, and robotic vision, often relying on deep learning models (such as Mask R-CNN) to achieve efficient pixel classification and instance differentiation through Convolutional Neural Networks (CNNs), providing powerful support for understanding complex scenes.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/01.png"/>
<b>The General Instance Segmentation Pipeline includes a</b> <b>Object Detection</b> <b>module. If you prioritize model precision, choose a model with higher precision. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Mask AP</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>Mask-RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>50.6</td>
<td>172.36 / 172.36</td>
<td>1615.75 / 1615.75</td>
<td>449.9 M</td>
<td rowspan="5">Mask-RT-DETR is an instance segmentation model based on RT-DETR. By adopting the high-performance PP-HGNetV2 as the backbone network and constructing a MaskHybridEncoder encoder, along with introducing IOU-aware Query Selection technology, it achieves state-of-the-art (SOTA) instance segmentation accuracy with the same inference time.</td>
</tr>
<tr>
<td>Mask-RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>45.7</td>
<td>88.18 / 88.18</td>
<td>1090.84 / 1090.84</td>
<td>113.6 M</td>
</tr>
</table>

> ❗ The above list features the <b>2 core models</b> that the image classification module primarily supports. In total, this module supports <b>15 models</b>. The complete list of models is as follows:

<details><summary> 👉Model List Details</summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Mask AP</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>36.3</td>
<td>141.69 / 141.69</td>
<td>nan / nan</td>
<td>254.8 M</td>
<td rowspan="2">Cascade-MaskRCNN is an improved Mask RCNN instance segmentation model that utilizes multiple detectors in a cascade, optimizing segmentation results by leveraging different IOU thresholds to address the mismatch between detection and inference stages, thereby enhancing instance segmentation accuracy.</td>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Trained Model</a></td>
<td>39.1</td>
<td>147.62 / 147.62</td>
<td>nan / nan</td>
<td>254.7 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>50.6</td>
<td>172.36 / 172.36</td>
<td>1615.75 / 1615.75</td>
<td>449.9 M</td>
<td rowspan="5">Mask-RT-DETR is an instance segmentation model based on RT-DETR. By adopting the high-performance PP-HGNetV2 as the backbone network and constructing a MaskHybridEncoder encoder, along with introducing IOU-aware Query Selection technology, it achieves state-of-the-art (SOTA) instance segmentation accuracy with the same inference time.</td>
</tr>
<tr>
<td>Mask-RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>45.7</td>
<td>88.18 / 88.18</td>
<td>1090.84 / 1090.84</td>
<td>113.6 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-M_pretrained.pdparams">Trained Model</a></td>
<td>42.7</td>
<td>78.69 / 78.69</td>
<td>nan / nan</td>
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
<td>114.16 / 114.16</td>
<td>1240.92 / 1240.92</td>
<td>237.5 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>35.6</td>
<td>118.30 / 118.30</td>
<td>nan / nan</td>
<td>157.5 M</td>
<td rowspan="6">Mask R-CNN is a full-task deep learning model from Facebook AI Research (FAIR) that can perform object classification and localization in a single model, combined with image-level masks to complete segmentation tasks.</td>
</tr>
<tr>
<td>MaskRCNN-ResNet50-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>36.4</td>
<td>118.34 / 118.34</td>
<td>nan / nan</td>
<td>157.5 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>32.8</td>
<td>228.83 / 228.83</td>
<td>nan / nan</td>
<td>128.7 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet101-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-FPN_pretrained.pdparams">Trained Model</a></td>
<td>36.6</td>
<td>148.14 / 148.14</td>
<td>nan / nan</td>
<td>225.4 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNet101-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>38.1</td>
<td>151.12 / 151.12</td>
<td>nan / nan</td>
<td>225.1 M</td>
</tr>
<tr>
<td>MaskRCNN-ResNeXt101-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNeXt101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>39.5</td>
<td>237.55 / 237.55</td>
<td>nan / nan</td>
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

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/02.png"/>

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the model within the pipeline</b>.

### 2.2 Local Experience
> ❗ Before using the general instance segmentation pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
* You can quickly experience the instance segmentation pipeline effect with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png), and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline instance_segmentation \
        --input general_instance_segmentation_004.png \
        --threshold 0.5 \
        --save_path ./output \
        --device gpu:0
```

The relevant parameter descriptions can be referred to in the parameter explanations in [2.2.2 Python Script Integration]().

After running, the result will be printed to the terminal as follows:

```bash
{'res': {'input_path': 'general_instance_segmentation_004.png', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'person', 'score': 0.8695873022079468, 'coordinate': [339.83426, 0, 639.8651, 575.22003]}, {'cls_id': 0, 'label': 'person', 'score': 0.8572642803192139, 'coordinate': [0.09976959, 0, 195.07274, 575.358]}, {'cls_id': 0, 'label': 'person', 'score': 0.8201770186424255, 'coordinate': [88.24664, 113.422424, 401.23077, 574.70197]}, {'cls_id': 0, 'label': 'person', 'score': 0.7110118269920349, 'coordinate': [522.54065, 21.457964, 767.5007, 574.2464]}, {'cls_id': 27, 'label': 'tie', 'score': 0.5543721914291382, 'coordinate': [247.38776, 312.4094, 355.2685, 574.1264]}], 'masks': '...'}}
```

The explanation of the result parameters can be referred to in [2.2.2 Python Script Integration](#222-python脚本方式集成).

The visualization results are saved under `save_path`, and the visualization results of instance segmentation are as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/03.png"/>

#### 2.2.2 Python Script Integration
* The above command line is for quickly experiencing and viewing the effect. Generally, in a project, it is often necessary to integrate through code. You can complete the quick inference of the production line with a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="instance_segmentation")
output = pipeline.predict(input="general_instance_segmentation_004.png", threshold=0.5)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps are performed:

(1) Instantiate the instance segmentation pipeline object through `create_pipeline()`, with specific parameter descriptions as follows:

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
<td>Pipeline name or path to pipeline config file, if it's set as a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
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
<td>Pipeline inference device. Supports specifying the specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU such as "cpu".</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the instance segmentation pipeline object for inference prediction. This method will return a `generator`. The parameters of the `predict()` method and their descriptions are as follows:

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
<li><b>str</b>: Local path of an image file or PDF file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as a network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code> (currently does not support prediction of directories containing PDF files, PDF files need to be specified to specific file paths)</li>
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
<td><code>threshold</code></td>
<td>Low score object filtering threshold for the model</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code> and less than <code>1</code>
<li><b>None</b>: If set to <code>None</code>, the default parameter <code>0.5</code> of the pipeline will be used as the threshold
</li></li></ul>

</td>
<td><code>None</code></td>

</table>

(3) Process the prediction results. The prediction result for each sample is of type `dict` and supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a json file</td>
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

    - `input_path`: `(str)` Input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page of the PDF; otherwise, it is `None`

    - `boxes`: `(list)` Detection box information, each element is a dictionary containing the following fields:
      - `cls_id`: `(int)` Class ID
      - `label`: `(str)` Class name
      - `score`: `(float)` Confidence of the detection box
      - `coordinate`: `(list)` Coordinates of the detection box, in the format [xmin, ymin, xmax, ymax]

    - `masks`: `...` The actual predicted mask of the instance segmentation model. Due to the large amount of data, it is not convenient to print directly, so it is replaced with `...`. You can save the prediction result as an image using `res.save_to_img` or save it as a json file using `res.save_to_json`.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.json`. If specified as a file, it will be saved directly to that file. Since json files do not support saving numpy arrays, `numpy.array` types will be converted to lists.

- Calling the `save_to_img()` method will save the visualization result to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If specified as a file, it will be saved directly to that file.

* In addition, you can also obtain the visualized image and prediction results through attributes, as follows:

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

- The prediction result obtained by the `json` attribute is a dictionary type, and the content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type. The key is `res`, and the corresponding value is an `Image.Image` object: an object used to display the prediction result of instance segmentation.

In addition, you can obtain the instance segmentation pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config instance_segmentation --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the instance segmentation production line by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the configuration file. An example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/instance_segmentation.yaml")

output = pipeline.predict(
    input="./general_instance_segmentation_004.png",
    threshold=0.5,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are the pipeline initialization parameters. If you wish to change the initialization parameters of the general instance segmentation pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline into your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python-script-method).

Moreover, PaddleX also provides three other deployment methods, as detailed below:

🚀 <b>High-Performance Inference</b>: In practical production environments, many applications have stringent performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX offers a high-performance inference plugin that deeply optimizes model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed information on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in practical production environments. By encapsulating inference functionality into services, clients can access these services via network requests to obtain inference results. PaddleX supports various pipeline service deployment solutions. For detailed information on pipeline service deployment, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
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
<p>Perform instance segmentation on an image.</p>
<p><code>POST /instance-segmentation</code></p>
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
<td>The URL of the image file accessible by the server or the Base64 encoded content of the image file.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following attributes:</li>
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
<td><code>instances</code></td>
<td><code>array</code></td>
<td>Information about the location, category, and other details of instances.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The result image of instance segmentation. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>instances</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>The location of the instance. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>The category ID of the instance.</td>
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
<p>The attributes of <code>mask</code> are as follows:</p>
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
<td><code>rleResult</code></td>
<td><code>str</code></td>
<td>The run-length encoding result of the mask.</td>
</tr>
<tr>
<td><code>size</code></td>
<td><code>array</code></td>
<td>The shape of the mask. The elements in the array are the height and width of the mask.</td>
</tr>
</tbody>
</table>
<p><code>result</code> example is as follows:</p>
<pre><code class="language-json">{
"instances": [
{
"bbox": [
162.39381408691406,
83.88176727294922,
624.0797119140625,
343.4986877441406
],
"categoryId": 33,
"score": 0.8691174983978271,
"mask": {
"rleResult": "xxxxxx",
"size": [
259,
462
]
}
}
],
"image": "xxxxxx"
}
</code></pre></details>
<details><summary>Multi-Language Service Call Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/instance-segmentation"  # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode the local image using Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nInstances:")
print(result["instances"])
</code></pre></details>
<details><summary>C++</summary>
<pre><code class="language-cpp">#include &lt;iostream&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string outputImagePath = "./out.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // Encode the local image using Base64
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; "Error reading file." &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post("/instance-segmentation", headers, body, "application/json");
    // Process the response data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; "Output image saved at " &lt;&lt; outputImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; "Unable to open file for writing: " &lt;&lt; outputImagePath &lt;&lt; std::endl;
        }

        auto instances = result["instances"];
        std::cout &lt;&lt; "\nInstances:" &lt;&lt; std::endl;
        for (const auto&amp; inst : instances) {
            std::cout &lt;&lt; inst &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; "Failed to send HTTP request." &lt;&lt; std::endl;
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
        String API_URL = "http://localhost:8080/instance-segmentation"; // Service URL
        String imagePath = "./demo.jpg"; // Local image
        String outputImagePath = "./out.jpg"; // Output image

        // Encode the local image to Base64
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
                JsonNode instances = result.get("instances");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                System.out.println("\nInstances: " + instances.toString());
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
    API_URL := "http://localhost:8080/instance-segmentation"
    imagePath := "./demo.jpg"
    outputImagePath := "./out.jpg"

    // Encode the local image in Base64
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64 encoded file content or image URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // Call the API
    client := &amp;http.Client{}
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

    // Process the returned data
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:"image"`
            Instances []map[string]interface{} `json:"instances"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
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
    fmt.Println("\nInstances:")
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
    static readonly string API_URL = "http://localhost:8080/instance-segmentation";
    static readonly string imagePath = "./demo.jpg";
    static readonly string outputImagePath = "./out.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local image using Base64
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64-encoded file content or image URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the API response data
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse["result"]["image"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output image saved at {outputImagePath}");
        Console.WriteLine("\nInstances:");
        Console.WriteLine(jsonResponse["result"]["instances"].ToString());
    }
}
</code></pre></details>
<details><summary>Node.js</summary>
<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/instance-segmentation';
const imagePath = './demo.jpg';
const outputImagePath = './out.jpg';

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64-encoded file content or image URL
  })
};

// Encode the local image using Base64
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) =&gt; {
    // Process the response data
    const result = response.data['result'];
    const imageBuffer = Buffer.from(result['image'], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log('\nInstances:');
    console.log(result['instances']);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>
<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/instance-segmentation"; // Service URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// Encode the local image using Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" =&gt; $image_data); // Base64-encoded file content or image URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the response data
$result = json_decode($response, true)['result'];
file_put_contents($output_image_path, base64_decode($result['image']));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nInstances:\n";
print_r($result['instances']);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data locally without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions on edge deployment, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs and proceed with the integration of AI applications.

## 4. Custom Development
If the default model weights provided by the general instance segmentation pipeline do not meet your requirements in terms of accuracy or speed, you can try to <b>further fine-tune the existing model using your own domain-specific or application-specific data</b> to improve the recognition performance of the general instance segmentation pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the general instance segmentation pipeline includes an instance segmentation module, if the pipeline's performance does not meet your expectations, you can analyze the images with poor segmentation results and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

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
<td>Prediction results are not as expected</td>
<td>Instance Segmentation Module</td>
<td><a href="../../../module_usage/tutorials/cv_modules/instance_segmentation.en.md">Link</a></td>
</tr>
</tbody>
</table>


### 4.2 Model Application
After completing the fine-tuning training with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, you only need to modify the pipeline configuration file by replacing the local path of the fine-tuned model weights in the corresponding position of the pipeline configuration file:

```yaml
SubModules:
  InstanceSegmentation:
    module_name: instance_segmentation
    model_name: Mask-RT-DETR-S
    model_dir: null # 替换为微调后的实例分割模型权重路径
    batch_size: 1
    threshold: 0.5
```

Subsequently, refer to the command-line or Python script methods in the local experience section to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for instance segmentation inference, the Python command is:

```bash
paddlex --pipeline instance_segmentation \
        --input general_instance_segmentation_004.png \
        --threshold 0.5 \
        --save_path ./output \
        --device npu:0
```

If you want to use the general instance segmentation production line on more types of hardware, please refer to [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
