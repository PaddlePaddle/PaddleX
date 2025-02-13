---
comments: true
---

# Seal Text Detection Module Development Tutorial

## I. Overview
The seal text detection module typically outputs multi-point bounding boxes around text regions, which are then passed as inputs to the distortion correction and text recognition modules for subsequent processing to identify the textual content of the seal. Recognizing seal text is an integral part of document processing and finds applications in various scenarios such as contract comparison, inventory access auditing, and invoice reimbursement verification. The seal text detection module serves as a subtask within OCR (Optical Character Recognition), responsible for locating and marking the regions containing seal text within an image. The performance of this module directly impacts the accuracy and efficiency of the entire seal text OCR system.

## II. Supported Model List


<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Hmean（%）</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109 M</td>
<td>The server-side seal text detection model of PP-OCRv4 boasts higher accuracy and is suitable for deployment on better-equipped servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6 M</td>
<td>The mobile-side seal text detection model of PP-OCRv4, on the other hand, offers greater efficiency and is suitable for deployment on end devices.</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation set for the above accuracy metrics is a self-built dataset containing 500 circular seal images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>


## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)


Just a few lines of code can complete the inference of the Seal Text Detection module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the Seal Text Detection module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png) to your local machine.

```python
from paddlex import create_model
model = create_model(model_name="PP-OCRv4_server_seal_det")
output = model.predict("seal_text_det.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the result is:

```bash
{'res': {'input_path': 'seal_text_det.png', 'dt_polys': [[[165, 469], [202, 500], [251, 523], [309, 535], [374, 527], [425, 506], [465, 475], [469, 473], [473, 473], [478, 476], [508, 506], [510, 510], [510, 514], [507, 521], [455, 561], [452, 562], [391, 586], [389, 587], [310, 597], [308, 597], [235, 583], [232, 583], [171, 554], [170, 552], [121, 510], [118, 506], [117, 503], [118, 498], [121, 496], [153, 469], [157, 466], [161, 466]], [[444, 444], [448, 447], [450, 450], [450, 453], [450, 497], [449, 501], [446, 503], [443, 505], [440, 506], [197, 506], [194, 505], [190, 503], [189, 499], [187, 493], [186, 490], [187, 453], [188, 449], [190, 446], [194, 444], [197, 443], [441, 443]], [[466, 346], [471, 350], [473, 351], [476, 356], [477, 361], [477, 425], [477, 430], [474, 434], [470, 437], [463, 439], [175, 440], [170, 439], [166, 437], [163, 432], [161, 426], [160, 361], [161, 357], [163, 352], [168, 349], [171, 347], [177, 345], [462, 345]], [[324, 38], [484, 92], [490, 95], [492, 97], [586, 227], [588, 231], [589, 236], [590, 384], [590, 390], [587, 394], [583, 397], [579, 399], [571, 398], [508, 379], [503, 377], [500, 374], [497, 369], [497, 366], [494, 260], [429, 170], [324, 136], [207, 173], [143, 261], [139, 366], [138, 370], [136, 375], [131, 378], [129, 379], [66, 397], [61, 397], [56, 397], [51, 393], [49, 390], [47, 383], [49, 236], [50, 230], [51, 227], [148, 96], [151, 92], [156, 90], [316, 38], [320, 37]]], 'dt_scores': [0.9929380286534535, 0.9980056201238314, 0.9936831226022099, 0.9884004535508197]}}
```

The meanings of the parameters are as follows:
- `input_path`: represents the path of the input image to be predicted
- `dt_polys`: represents the predicted text detection boxes, where each text detection box contains multiple vertices of a polygon. Each vertex is a tuple of two elements, representing the x and y coordinates of the vertex respectively
- `dt_scores`: represents the confidence scores of the predicted text detection boxes

The visualization image is as follows:

<img alt="Visualization Image" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/seal_text_det/seal_text_det_res.png"/>

The explanations of related methods and parameters are as follows:

* `create_model` instantiates a text detection model (here we take `PP-OCRv4_server_seal_det` as an example), and the specific explanations are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>All model names supported by PaddleX for seal text detection</td>
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
<td><code>limit_side_len</code></td>
<td>Limit on the side length of the image for detection</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the default value from the official PaddleX model configuration will be used</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Type of side length limit for detection</td>
<td><code>str/None</code></td>
<td>
<ul>
<li><b>str</b>: Supports min and max. min ensures the shortest side of the image is not less than det_limit_side_len, max ensures the longest side is not greater than limit_side_len
<li><b>None</b>: If set to None, the default value from the official PaddleX model configuration will be used</li></li></ul></td>


<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>In the output probability map, pixels with scores greater than this threshold will be considered as text pixels</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the default value from the official PaddleX model configuration will be used</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>If the average score of all pixels within a detection result box is greater than this threshold, the result will be considered as a text region</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the default value from the official PaddleX model configuration will be used</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>max_candidates</code></td>
<td>Maximum number of text boxes to output</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the default value from the official PaddleX model configuration will be used</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Expansion ratio for the Vatti clipping algorithm, used to expand the text region</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the default value from the official PaddleX model configuration will be used</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>use_dilation</code></td>
<td>Whether to dilate the segmentation result</td>
<td><code>bool/None</code></td>
<td>True/False/None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the built-in model parameters of PaddleX will be used by default. On this basis, if `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the seal text detection model is called for inference prediction. The parameters of the `predict()` method include `input`, `batch_size`, `limit_side_len`, `limit_type`, `thresh`, `box_thresh`, `max_candidates`, `unclip_ratio`, and `use_dilation`. The specific descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL Link</b>, such as the web URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
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
<td><code>limit_side_len</code></td>
<td>Side length limit for detection</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>limit_type</code></td>
<td>Type of side length limit for detection</td>
<td><code>str/None</code></td>
<td>
<ul>
<li><b>str</b>: Supports min and max. min indicates that the shortest side of the image is not less than det_limit_side_len, max indicates that the longest side of the image is not greater than limit_side_len
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>


<td>None</td>
</tr>
<tr>
<td><code>thresh</code></td>
<td>In the output probability map, pixels with scores greater than this threshold will be considered as text pixels</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>box_thresh</code></td>
<td>If the average score of all pixels within the detection result box is greater than this threshold, the result will be considered as a text area</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>max_candidates</code></td>
<td>Maximum number of text boxes to be output</td>
<td><code>int/None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>unclip_ratio</code></td>
<td>Expansion coefficient of the Vatti clipping algorithm, used to expand the text area</td>
<td><code>float/None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0
<li><b>None</b>: If set to None, the parameter value initialized by the model will be used by default</li></li></ul></td>

<td>None</td>
</tr>
<tr>
<td><code>use_dilation</code></td>
<td>Whether to dilate the segmentation result</td>
<td><code>bool/None</code></td>
<td>True/False/None</td>
<td>None</td>
</tr>
</table>

* Process the prediction results. Each sample's prediction result is a corresponding Result object, and it supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a file in JSON format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as a file in image format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining visual images with results and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visual image in <code>dict</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development

If you seek higher accuracy, you can leverage PaddleX's custom development capabilities to develop better Seal Text Detection models. Before developing a Seal Text Detection model with PaddleX, ensure you have installed PaddleOCR plugin for PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Dataset Preparation

Before model training, you need to prepare a dataset for the task. PaddleX provides data validation functionality for each module. <b>Only data that passes validation can be used for model training.</b> Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Text Detection and Recognition Task Module Data Preparation Tutorial](../../../data_annotations/ocr_modules/text_detection_recognition.en.md).

#### 4.1.1 Demo Data Download

You can download the demo dataset to a specified folder using the following commands:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_curve_det_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_curve_det_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Once the command runs successfully, a message saying `Check dataset passed !` will be printed in the log. The verification results will be saved in `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory, including visual examples of sample images and a histogram of sample distribution.


<details><summary>👉 <b>Verification Result Details (click to expand)</b></summary>
<p>The specific content of the verification result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 606,
    "train_sample_paths": [
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug07834.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug09943.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug04079.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug05701.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug08324.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug07451.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug09562.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug08237.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug01788.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug06481.png"
    ],
    "val_samples": 152,
    "val_sample_paths": [
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug03724.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug06456.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug04029.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug03603.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug05454.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug06269.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug00624.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug02818.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug00538.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug04935.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": ".\/ocr_curve_det_dataset_examples",
  "show_type": "image",
  "dataset_type": "TextDetDataset"
}
</code></pre>
<p>The verification results above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 606;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 152;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualization images of validation samples in this dataset;</li>
</ul>
<p>The dataset verification also analyzes the distribution of sample numbers across all classes and plots a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/curved_text_dec/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>
<p>After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by modifying the configuration file or appending hyperparameters.</p>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Seal text detection does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> fields in the configuration file. Example explanations for some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to enable re-splitting the dataset, set to <code>True</code> to perform dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set, which should be an integer between 0 and 100, ensuring the sum with <code>val_percent</code> is 100;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training

Model training can be completed with just one command. Here, we use the Seal Text Detection model (PP-OCRv4_server_seal_det) as an example:

```bash
python main.py -c paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `PP-OCRv4_server_seal_det.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to train using the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters Documentation](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves model weight files, with the default path being <code>output</code>. To specify a different save path, use the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights are produced, and static graph weights are used by default for model inference.</li>
<li>
<p>After model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, including whether the training task completed successfully, produced weight metrics, and related file paths.</p>
</li>
<li><code>train.log</code>: Training log file, recording model metric changes, loss changes, etc.</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameters used for this training session.</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, and static graph network structure.</li>
</ul></details>

### 4.3 Model Evaluation
After model training, you can evaluate the specified model weights on the validation set to verify model accuracy. Using PaddleX for model evaluation requires just one command:

```bash
python main.py -c paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
```

Similar to model training, follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `PP-OCRv4_server_seal_det.yaml`).
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the validation dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For more details, refer to the [PaddleX Common Configuration Parameters Documentation](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter, e.g., <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After model evaluation, the following outputs are typically produced:</p>
<ul>
<li><code>evaluate_result.json</code>: Records the evaluation results, specifically whether the evaluation task completed successfully and the model's evaluation metrics, including precision, recall and Hmean.</li>
</ul></details>

### 4.4 Model Inference and Integration
After model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions via the command line, use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png) to your local machine.


```bash
python main.py -c paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="seal_text_det.png"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `PP-OCRv4_server_seal_det.yaml`)

* Set the mode to model inference prediction: `-o Global.mode=predict`

* Specify the model weights path: -o Predict.model_dir="./output/best_accuracy/inference"

Specify the input data path: `-o Predict.inputh="..."` Other related parameters can be set by modifying the fields under Global and Predict in the `.yaml` configuration file. For details, refer to PaddleX Common Model Configuration File Parameter Description.

Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration

The model can be directly integrated into the PaddleX pipeline or into your own projects.

1. <b>Pipeline Integration</b>

The document Seal Text Detection module can be integrated into PaddleX pipelines such as the [General OCR Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/OCR.en.md) and [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3-doc)](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.en.md). Simply replace the model path to update the text detection module of the relevant pipeline.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the Seal Text Detection module. You can refer to the Python sample code in [Quick Integration](#iii-quick-integration) and just replace the model with the path to the model you trained.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).