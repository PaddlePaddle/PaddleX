---
comments: true
---

# Layout Detection Module Development Tutorial

## I. Overview
The core task of structure analysis is to parse and segment the content of input document images. By identifying different elements in the image (such as text, charts, images, etc.), they are classified into predefined categories (e.g., pure text area, title area, table area, image area, list area, etc.), and the position and size of these regions in the document are determined.

## II. Supported Model List

* <b>The layout detection model includes 23 common categories: document title, paragraph title, text, page number, abstract, table of contents, references, footnotes, header, footer, algorithm, formula, formula number, image, figure caption, table, table caption, seal, figure title, figure, header image, footer image, and sidebar text</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.5252</td>
<td>1454.27</td>
<td>123.76 M</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>15.9</td>
<td>160.1</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>13.8</td>
<td>46.7</td>
<td>4.834</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-S.</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 500 common document-type images of Chinese and English papers, magazines, contracts, books, exams, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

> ❗ The above list includes the <b>3 core models</b> that are key supported by the text recognition module. The module actually supports a total of <b>11 full models</b>, including several predefined models with different categories. The complete model list is as follows:

<details><summary> 👉 Details of Model List</summary>

* <b>Table Layout Detection Model</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset using PicoDet-1x, capable of detecting table regions.</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout table area detection dataset by PaddleOCR, containing 7835 Chinese and English document images with tables. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>3-Class Layout Detection Model, including Table, Image, and Stamp</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H.</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 1154 common document images of Chinese and English papers, magazines, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>5-Class English Document Area Detection Model, including Text, Title, Table, Image, and List</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>A high-efficiency English document layout area localization model trained on the PubLayNet dataset using PicoDet-1x.</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above precision metrics is the [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) dataset, containing 11245 English document images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>17-Class Area Detection Model, including 17 common layout categories: Paragraph Title, Image, Text, Number, Abstract, Content, Figure Caption, Formula, Table, Table Caption, References, Document Title, Footnote, Header, Algorithm, Footer, and Stamp</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H.</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 892 common document images of Chinese and English papers, magazines, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>
</details>


## III. Quick Integration  <a id="quick"> </a>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Tutorial](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the structure analysis module. You can switch models under this module freely, and you can also integrate the model inference of the structure analysis module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "PP-DocLayout-L"
model = create_model(model_name=model_name)
output = model.predict("layout.jpg", batch_size=1, layout_nms=True)

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")

```

<details><summary>👉 <b>After running, the result is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'layout.jpg', 'page_index': None, 'boxes': [{'cls_id': 8, 'label': 'table', 'score': 0.9866330623626709, 'coordinate': [74.30875, 105.7123, 321.98837, 299.10858]}, {'cls_id': 2, 'label': 'text', 'score': 0.9859815835952759, 'coordinate': [34.658115, 349.9102, 358.33835, 611.34314]}, {'cls_id': 2, 'label': 'text', 'score': 0.9850621223449707, 'coordinate': [34.944622, 647.37946, 358.3256, 849.2333]}, {'cls_id': 8, 'label': 'table', 'score': 0.9850116968154907, 'coordinate': [438.0683, 105.377174, 662.8872, 313.88733]}, {'cls_id': 2, 'label': 'text', 'score': 0.9847850799560547, 'coordinate': [385.97052, 497.03925, 710.9549, 697.6805]}, {'cls_id': 2, 'label': 'text', 'score': 0.9805687665939331, 'coordinate': [385.79697, 345.93753, 710.07336, 459.14374]}, {'cls_id': 2, 'label': 'text', 'score': 0.9799897074699402, 'coordinate': [386.07574, 735.3798, 710.60785, 850.2002]}, {'cls_id': 9, 'label': 'table_title', 'score': 0.9375936388969421, 'coordinate': [35.273827, 19.851961, 358.92508, 77.812584]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8755733966827393, 'coordinate': [386.63184, 476.60715, 699.78345, 490.1164]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8617452383041382, 'coordinate': [387.27448, 715.95795, 524.38367, 729.20795]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8607752919197083, 'coordinate': [35.451626, 627.496, 185.63443, 640.4029]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8575425148010254, 'coordinate': [35.33491, 330.8047, 141.46854, 344.4073]}, {'cls_id': 9, 'label': 'table_title', 'score': 0.7957680225372314, 'coordinate': [385.9386, 19.754547, 711.51184, 75.00631]}]}}
```
The meanings of the parameters are as follows:
- `input_path`: The path to the input image for prediction.
- `page_index`: If the input is a PDF file, it indicates which page of the PDF it is; otherwise, it is `None`.
- `boxes`: Information about the predicted bounding boxes, a list of dictionaries. Each dictionary represents a detected object and contains the following information:
  - `cls_id`: Class ID, an integer.
  - `label`: Class label, a string.
  - `score`: Confidence score of the bounding box, a float.
  - `coordinate`: Coordinates of the bounding box, a list of floats in the format <code>[xmin, ymin, xmax, ymax]</code>.

</details>

The visualized image is as follows:
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/layout_det/layout_res.jpg"/>

Relevant methods, parameters, and explanations are as follows:

* `create_model` instantiates a target detection model (here, `PP-DocLayout-L` is used as an example). The detailed explanation is as follows:
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
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>Size of the input image; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>int/list/None</code></td>
<td>
<ul>
<li><b>int</b>, e.g., 640, means resizing the input image to 640x640</li>
<li><b>List</b>, e.g., [640, 512], means resizing the input image to a width of 640 and a height of 512</li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, e.g., 0.2, means filtering out all bounding boxes with a confidence score less than 0.2</li>
<li><b>Dictionary</b>, with keys as <b>int</b> representing <code>cls_id</code> and values as <b>float</b> thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> means applying a threshold of 0.45 for cls_id 0, 0.48 for cls_id 2, and 0.4 for cls_id 7</li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping boxes; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>bool/None</code></td>
<td>
<ul>
<li><b>bool</b>, True/False, indicates whether to use NMS for post-processing to filter overlapping boxes</li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of the detection box; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float/list/None</code></td>
<td>
<ul>
<li><b>float</b>, a positive float number, e.g., 1.1, means expanding the width and height of the detection box by 1.1 times while keeping the center unchanged</li>
<li><b>List</b>, e.g., [1.2, 1.5], means expanding the width by 1.2 times and the height by 1.5 times while keeping the center unchanged</li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for the detection boxes output by the model; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>string/None</code></td>
<td>
<ul>
<li><b>large</b>, when set to large, only the largest external box will be retained for overlapping detection boxes, and the internal overlapping boxes will be deleted</li>
<li><b>small</b>, when set to small, only the smallest internal box will be retained for overlapping detection boxes, and the external overlapping boxes will be deleted</li>
<li><b>union</b>, no filtering of boxes will be performed, and both internal and external boxes will be retained</li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
</tr></table>

* Note that `model_name` must be specified. After specifying `model_name`, the default PaddleX built-in model parameters will be used. If `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the target detection model is called for inference prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `threshold`, which are explained as follows:

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
<td>Data for prediction, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL link</b>, such as the network URL of an image file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg">示例</a></li>
<li><b>Local Directory</b>, the directory should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, the elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer greater than 0</td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, e.g., 0.2, means filtering out all bounding boxes with a confidence score less than 0.2</li>
<li><b>Dictionary</b>, with keys as <b>int</b> representing <code>cls_id</code> and values as <b>float</b> thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> means applying a threshold of 0.45 for cls_id 0, 0.48 for cls_id 2, and 0.4 for cls_id 7</li>
<li><b>None</b>, not specified, will use the <code>threshold</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default PaddleX official model configuration will be used</li>
</ul>
</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping boxes; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>bool/None</code></td>
<td>
<ul>
<li><b>bool</b>, True/False, indicates whether to use NMS for post-processing to filter overlapping boxes</li>
<li><b>None</b>, not specified, will use the <code>layout_nms</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of the detection box; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float/list/None</code></td>
<td>
<ul>
<li><b>float</b>, a positive float number, e.g., 1.1, means expanding the width and height of the detection box by 1.1 times while keeping the center unchanged</li>
<li><b>List</b>, e.g., [1.2, 1.5], means expanding the width by 1.2 times and the height by 1.5 times while keeping the center unchanged</li>
<li><b>None</b>, not specified, will use the <code>layout_unclip_ratio</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for the detection boxes output by the model; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>string/None</code></td>
<td>
<ul>
<li><b>large</b>, when set to large, only the largest external box will be retained for overlapping detection boxes, and the internal overlapping boxes will be deleted</li>
<li><b>small</b>, when set to small, only the smallest internal box will be retained for overlapping detection boxes, and the external overlapping boxes will be deleted</li>
<li><b>union</b>, no filtering of boxes will be performed, and both internal and external boxes will be retained</li>
<li><b>None</b>, not specified, will use the <code>layout_merge_bboxes_mode</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td>None</td>
</tr>
</tr></table>

* Additionally, it also supports obtaining the visualized image with results and the prediction results via attributes, as follows:

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
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference API, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better structure analysis models. Before developing a structure analysis model with PaddleX, ensure you have installed PaddleX's Detection-related model training capabilities. The installation process can be found in [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development based on the official demos. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_layout_examples.tar -P ./dataset
tar -xf ./dataset/det_layout_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_layout_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 11,
    "train_samples": 90,
    "train_sample_paths": [
      "check_dataset/demo_img/JPEGImages/train_0077.jpg",
      "check_dataset/demo_img/JPEGImages/train_0028.jpg",
      "check_dataset/demo_img/JPEGImages/train_0012.jpg"
    ],
    "val_samples": 20,
    "val_sample_paths": [
      "check_dataset/demo_img/JPEGImages/val_0007.jpg",
      "check_dataset/demo_img/JPEGImages/val_0019.jpg",
      "check_dataset/demo_img/JPEGImages/val_0010.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "det_layout_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
</code></pre>
<p>The verification results mentioned above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Details of other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 11;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 90;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 20;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of validation samples in this dataset;</li>
</ul>
<p>The dataset verification also analyzes the distribution of sample numbers across all classes and generates a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/layout_dec/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Layout detection does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> section in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, ensuring the sum with <code>val_percent</code> is 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_layout_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code>python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_layout_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training

A single command is sufficient to complete model training, taking the training of PP-DocLayout-L as an example:

```bash
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/det_layout_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file of the model (here it is `PP-DocLayout-L.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves model weight files, defaulting to <code>output</code>. To specify a save path, use the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics and loss during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configuration for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

### <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/det_layout_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PP-DocLayout-L.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="layout.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PP-DocLayout-L.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

* Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own project. To integrate, simply add the `model_dir="/output/best_model/inference"` parameter to the `create_model(model_name=model_name, kernel_option=kernel_option)` function in the quick integration method from Step 3.

#### 4.4.2 Model Integration
The model can be directly integrated into PaddleX pipelines or into your own projects.

1. <b>Pipeline Integration</b>
The structure analysis module can be integrated into PaddleX pipelines such as the [General Table Recognition Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition.en.md) and the [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3-doc)](../../..//pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.en.md). Simply replace the model path to update the layout area localization module. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

1. <b>Module Integration</b>
The weights you produce can be directly integrated into the layout area localization module. You can refer to the Python example code in the [Quick Integration](#quick) section, simply replacing the model with the path to your trained model.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).