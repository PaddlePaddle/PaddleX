---
comments: true
---

# Table Cell Detection Module Usage Guide

## I. Overview
The table cell detection module is a key component of table recognition tasks, responsible for locating and marking each cell area in table images. The performance of this module directly affects the accuracy and efficiency of the entire table recognition process. The table cell detection module typically outputs bounding boxes for each cell area, which will be passed as input to the table recognition pipeline for subsequent processing.

## II. List of Supported Models

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. The Baidu PaddlePaddle Vision Team, based on RT-DETR-L as the base model, has completed pretraining on a self-built table cell detection dataset, achieving good performance for both wired and wireless table cell detection.
</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>

<p><b>Note: The above accuracy metrics are measured from the internal table cell detection dataset of PaddleX. All model GPU inference times are based on an NVIDIA Tesla T4 machine, with precision type FP32. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.</b></p>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package first. For details, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can complete the inference of the table cell detection module with just a few lines of code. You can switch between the models under this module at will, and you can also integrate the model inference of the table cell detection module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your local machine.

```python
from paddlex import create_model
model = create_model(model_name="RT-DETR-L_wired_table_cell_det")
output = model.predict("table_recognition.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

<details><summary>👉 <b>After running, the result is: (Click to expand)</b></summary>

```json
{'res': {'input_path': 'table_recognition.jpg', 'boxes': [{'cls_id': 0, 'label': 'cell', 'score': 0.9319108128547668, 'coordinate': [109.835846, 95.89979, 212.7077, 127.055466]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9308021664619446, 'coordinate': [109.75361, 64.866486, 212.84799, 95.822426]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9255117177963257, 'coordinate': [110.00513, 30.894377, 212.81178, 64.80416]}, {'cls_id': 0, 'label': 'cell', 'score': 0.918117344379425, 'coordinate': [212.87247, 30.97587, 403.8024, 64.86235]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9053983688354492, 'coordinate': [212.89151, 95.95629, 403.36572, 127.11717]}, {'cls_id': 0, 'label': 'cell', 'score': 0.8567661046981812, 'coordinate': [212.77899, 64.98128, 403.9478, 95.87939]}, {'cls_id': 0, 'label': 'cell', 'score': 0.7800847887992859, 'coordinate': [404.12827, 64.99693, 547.1579, 95.95234]}, {'cls_id': 0, 'label': 'cell', 'score': 0.7557389736175537, 'coordinate': [2.657493, 30.968334, 109.947815, 64.894485]}, {'cls_id': 0, 'label': 'cell', 'score': 0.6763500571250916, 'coordinate': [2.5346346, 96.218285, 109.79284, 127.097565]}, {'cls_id': 0, 'label': 'cell', 'score': 0.6708637475967407, 'coordinate': [404.02423, 95.9553, 547.27985, 127.17637]}, {'cls_id': 0, 'label': 'cell', 'score': 0.6568276286125183, 'coordinate': [2.2822304, 65.10485, 109.99168, 95.964096]}, {'cls_id': 0, 'label': 'cell', 'score': 0.6159431338310242, 'coordinate': [109.78963, 95.94173, 213.05418, 127.06708]}, {'cls_id': 0, 'label': 'cell', 'score': 0.6098588109016418, 'coordinate': [2.2127364, 65.04467, 110.07493, 95.99106]}, {'cls_id': 0, 'label': 'cell', 'score': 0.6019916534423828, 'coordinate': [403.98883, 96.003845, 547.2072, 127.17022]}, {'cls_id': 0, 'label': 'cell', 'score': 0.5713056921958923, 'coordinate': [404.4564, 30.951345, 547.1254, 65.081154]}, {'cls_id': 0, 'label': 'cell', 'score': 0.5697788000106812, 'coordinate': [212.81021, 96.05031, 403.7318, 127.14639]}]}}
```

<details><summary>👉 <b>After running, the result is: (Click to expand)</b></summary>

The meanings of the parameters are as follows:
- `input_path`: The path of the input image to be predicted.
- `boxes`: Information of the predicted bounding boxes, a list of dictionaries. Each dictionary represents a detected target, containing the following information:
  - `cls_id`: Class ID, an integer.
  - `label`: Class label, a string.
  - `score`: Confidence of the bounding box, a floating-point number.
  - `coordinate`: Coordinates of the bounding box, a list of floating-point numbers, in the format <code>[xmin, ymin, xmax, ymax]</code>

</details>

The visualized image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/table_cells_det/table_cells_det_res.jpg">

Note: Due to network issues, the above URL may not be successfully parsed. If you need the content from this link, please check the validity of the URL and try again. If you do not need the content from this link, please let me know, and I will proceed with answering your question.

The following is the explanation of the methods, parameters, etc.:

* The `create_model` method instantiates a table cell detection model (here, `RT-DETR-L_wired_table_cell_det` is used as an example), with the following details:
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
<td>Size of the input image; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>int/list</code></td>
<td>
<ul>
  <li><b>int</b>, e.g., 640, means resizing the input image to 640x640</li>
  <li><b>List</b>, e.g., [640, 512], means resizing the input image to a width of 640 and a height of 512</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold to filter out low-confidence predictions; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>float/dict</code></td>
<td>
<ul>
  <li><b>float</b>, e.g., 0.2, means filtering out all bounding boxes with a confidence score less than 0.2</li>
  <li><b>Dictionary</b>, with keys as <b>int</b> representing <code>cls_id</code> and values as <b>float</b> thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> means applying a threshold of 0.45 for class ID 0, 0.48 for class ID 2, and 0.4 for class ID 7</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. Once `model_name` is specified, the default model parameters from PaddleX will be used. If `model_dir` is specified, the user-defined model will be used instead.

* The `predict()` method of the table cell detection model is called to perform inference and prediction. The parameters of the `predict()` method include `input`, `batch_size`, and `threshold`, with the following details:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Optional</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python variable</b>, such as <code>numpy.ndarray</code> representing image data</li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as the network URL of an image file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">Example</a></li>
  <li><b>Local directory</b>, the directory must contain files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>Dictionary</b>, the <code>key</code> of the dictionary must correspond to the specific task, such as <code>"img"</code> for image classification tasks, and the <code>val</code> of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code></li>
  <li><b>List</b>, elements of the list must be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer</td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering out low-confidence prediction results; if not specified, the <code>threshold</code> parameter specified in <code>creat_model</code> will be used by default, and if <code>creat_model</code> also does not specify it, the default PaddleX official model configuration will be used</td>
<td><code>float/dict</code></td>
<td>
<ul>
  <li><b>float</b>, such as 0.2, indicating that all target boxes with a threshold less than 0.2 will be filtered out</li>
  <li><b>Dictionary</b>, the key of the dictionary is of type <b>int</b>, representing <code>cls_id</code>, and the val is a <b>float</b> type threshold. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>, indicating that a threshold of 0.45 is applied to the class with cls_id 0, a threshold of 0.48 is applied to the class with cls_id 1, and a threshold of 0.4 is applied to the class with cls_id 7</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* Process the prediction results, where the prediction result of each sample is a corresponding Result object, and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save to, and when it is a directory, the saved file name is consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save to, and when it is a directory, the saved file name is consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining visualized images with results and prediction results through attributes, as follows:
  
<table>
<thead>
<tr>
<th>Property</th>
<th>Property Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more information on the usage of PaddleX single-model inference APIs, please refer to [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Secondary Development
If you aim to improve the accuracy of existing models, you can leverage PaddleX's secondary development capabilities to develop a better table cell detection model. Before using PaddleX to develop a table cell detection model, please ensure that the PaddleX table cell detection model training plugin is installed. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before training the model, you need to prepare the dataset for the corresponding task module. PaddleX provides a data validation feature for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX offers demo datasets for each module, which you can use to complete subsequent development based on the official demo data. If you wish to use your private dataset for model training, please refer to the [PaddleX Object Detection Task Module Data Annotation Guide](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can use the following command to download the demo dataset to the specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cells_det_coco_examples.tar -P ./dataset
tar -xf ./dataset/cells_det_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cells_det_coco_examples
```

After executing the above command, PaddleX will verify the dataset and collect basic information about the dataset. If the command runs successfully, it will print `Check dataset passed !` in the log. The verification result file is saved in `./output/check_dataset_result.json`, and the related outputs will be saved in the `./output/check_dataset` directory under the current directory. The output directory includes visualized example sample images and sample distribution histograms.

<details><summary>👉 <b>Verification Result Details (Click to Expand)</b></summary>

<p>The specific content of the verification result file is as follows:</p>

```json
"done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 1,
    "train_samples": 230,
    "train_sample_paths": [
      "check_dataset\/demo_img\/img_45_2.png",
      "check_dataset\/demo_img\/img_69_1.png",
      "check_dataset\/demo_img\/img_99_1.png",
      "check_dataset\/demo_img\/img_6_1.png",
      "check_dataset\/demo_img\/img_47_3.png",
      "check_dataset\/demo_img\/img_54_2.png",
      "check_dataset\/demo_img\/img_25_1.png",
      "check_dataset\/demo_img\/img_73_1.png",
      "check_dataset\/demo_img\/img_51_2.png",
      "check_dataset\/demo_img\/img_93_3.png"
    ],
    "val_samples": 26,
    "val_sample_paths": [
      "check_dataset\/demo_img\/img_88_2.png",
      "check_dataset\/demo_img\/img_156_0.png",
      "check_dataset\/demo_img\/img_43_4.png",
      "check_dataset\/demo_img\/img_2_4.png",
      "check_dataset\/demo_img\/img_42_4.png",
      "check_dataset\/demo_img\/img_49_0.png",
      "check_dataset\/demo_img\/img_45_1.png",
      "check_dataset\/demo_img\/img_140_0.png",
      "check_dataset\/demo_img\/img_5_1.png",
      "check_dataset\/demo_img\/img_26_3.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "cells_det_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
```

<p>In the above verification results, <code>check_pass</code> being true indicates that the dataset format meets the requirements. The explanations for other metrics are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 1;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 230;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 26;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualization images of the training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualization images of the validation samples in this dataset;</li>
</ul>
<p>In addition, the dataset verification has analyzed the distribution of sample counts for all classes in the dataset and plotted a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/table_cells_det/01.png"></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After you complete the data verification, you can convert the dataset format by <b>modifying the configuration file</b> or <b>adding hyperparameters</b>. You can also re-split the training/validation ratio of the dataset.

<details><summary>👉 <b>Details of Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Table cell detection supports converting datasets in <code>VOC</code> and <code>LabelMe</code> formats to the <code>COCO</code> format.</p>
<p>Parameters related to dataset verification can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example explanations of parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion. Table cell detection supports converting datasets in <code>VOC</code> and <code>LabelMe</code> formats to the <code>COCO</code> format. The default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If performing dataset format conversion, the source dataset format must be set. The default is <code>null</code>, and the available options are <code>VOC</code>, <code>LabelMe</code>, <code>VOCWithUnlabeled</code>, and <code>LabelMeWithUnlabeled</code>. For example, if you want to convert a dataset in <code>LabelMe</code> format to <code>COCO</code> format, using the following <code>LabelMe</code> dataset as an example, you need to modify the configuration as follows:</li>
</ul>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_labelme_examples.tar -P ./dataset
tar -xf ./dataset/det_labelme_examples.tar -C ./dataset/
</code></pre>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples
</code></pre>
<p>Of course, the above parameters can also be set by adding command-line arguments. For example, for a dataset in <code>LabelMe</code> format:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example explanations of parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. It is set to <code>True</code> when performing dataset format conversion. The default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, you need to set the percentage of the training set. It is an integer between 0 and 100, and it must sum up to 100 with the value of <code>val_percent</code>;</li>
<li><code>val_percent</code>: If re-splitting the dataset, you need to set the percentage of the validation set. It is an integer between 0 and 100, and it must sum up to 100 with the value of <code>train_percent</code>. For example, if you want to re-split the dataset with 90% for training and 10% for validation, you need to modify the configuration file as follows:</li>
</ul>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cells_det_coco_examples
</code></pre>
<p>After the dataset splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by adding command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cells_det_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
A single command can complete the model training. Taking the table cell detection model `RT-DETR-L_wired_table_cell_det` as an example:

```bash
python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cells_det_coco_examples
```

The following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `RT-DETR-L_wired_table_cell_det.yaml`. When training other models, the corresponding configuration file must be specified. The correspondence between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or by adding parameters in the command line. For example, to train on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding model task module in [PaddleX Common Model Configuration Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves model weight files, with the default directory being <code>output</code>. If you need to specify a different save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts away the concepts of dynamic graph weights and static graph weights for you. During model training, both dynamic and static graph weights are generated. By default, static graph weights are used for model inference.</li>
<li>
<p>After model training is completed, all outputs are saved in the specified output directory (default is <code>./output/</code>), and typically include the following:</p>
</li>
<li>
<p><code>train_result.json</code>: The training result record file, which logs whether the training task was completed normally, as well as the metrics of the generated weights and related file paths;</p>
</li>
<li><code>train.log</code>: The training log file, which records changes in model metrics and loss during the training process;</li>
<li><code>config.yaml</code>: The training configuration file, which logs the hyperparameter settings for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: These are model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, and static graph network structure, etc.</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cells_det_coco_examples
```

Similar to model training, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `RT-DETR-L_wired_table_cell_det.yaml`).
* Set the mode to model evaluation: `-o Global.mode=evaluate`.
* Specify the validation dataset path: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX General Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

<p>When evaluating the model, the path to the model weights file needs to be specified. Each configuration file has a default weight save path built-in. If you need to change it, you can simply set it by appending a command-line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After the model evaluation is completed, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed normally and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing the training and evaluation of the model, you can use the trained model weights for inference prediction or integrate them into Python.

#### 4.4.1 Model Inference

* To perform inference prediction via the command line, you only need the following command. Before running the code below, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your local machine. Note: The link may not work due to network issues. If you encounter problems, please check the validity of the link and try again.

Similar to model training and evaluation, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `RT-DETR-L_wired_table_cell_det.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`

Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX General Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX production line or into your own project.

1.<b>Production Line Integration</b>

The table cell detection module can be integrated into the PaddleX production line [General Table Recognition Production Line v2](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.en.md). Simply replacing the model path will complete the model update for the table cell detection module in the relevant production line. In production line integration, you can deploy your model using high-performance deployment and service-oriented deployment.

2.<b>Module Integration</b>

The weights you generate can be directly integrated into the table cell detection module. You can refer to the Python example code in [Quick Integration](#3-Quick-Integration). Simply replace the model with the path of the model you have trained.