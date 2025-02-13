---
comments: true
---

# Mainbody detection Module Development Tutorial

## I. Overview
Mainbody detection is a fundamental task in object detection, aiming to identify and extract the location and size of specific target objects, people, or entities from images and videos. By constructing deep neural network models, mainbody detection learns the feature representations of image subjects to achieve efficient and accurate detection.

## II. Supported Model List


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-ShiTuV2_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_det_pretrained.pdparams">Trained Model</a></td>
<td>41.5</td>
<td>62.0</td>
<td>12.79 / 4.51</td>
<td>44.14 / 44.14</td>
<td>27.54</td>
<td>A mainbody detection model based on PicoDet_LCNet_x2_5, which may detect multiple common subjects simultaneously.</td>
</tr>
</table>
<b>Note: The evaluation set for the above accuracy metrics is  PaddleClas mainbody detection dataset mAP(0.5:0.95). GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## III. Quick Integration  <a id="quick"> </a>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, you can perform mainbody detection inference with just a few lines of code. You can easily switch between models under this module, and integrate the mainbody detection model inference into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png) to your local machine.

```python
from paddlex import create_model
model_name = "PP-ShiTuV2_det"
model = create_model(model_name)
output = model.predict("general_object_detection_002.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```bash
{'res': "{'input_path': 'general_object_detection_002.png', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'mainbody', 'score': 0.8161919713020325, 'coordinate': [76.07117, 272.83017, 329.5627, 519.48236]}, {'cls_id': 0, 'label': 'mainbody', 'score': 0.8071584701538086, 'coordinate': [662.7539, 92.804276, 874.7139, 308.21216]}, {'cls_id': 0, 'label': 'mainbody', 'score': 0.754974365234375, 'coordinate': [284.4833, 93.76895, 476.6789, 297.27588]}, {'cls_id': 0, 'label': 'mainbody', 'score': 0.6657832860946655, 'coordinate': [732.1591, 0, 1035.9547, 168.45923]}, {'cls_id': 0, 'label': 'mainbody', 'score': 0.614763081073761, 'coordinate': [763.9127, 280.74258, 925.48065, 439.444]}, ... ]}"}
```

The meanings of the running results parameters are as follows:
- `input_path`: Indicates the path of the input image to be predicted.
- `page_index`: If the input is a PDF file, it represents the current page number of the PDF; otherwise, it is `None`.
- `boxes`: Information of each detected object.
  - `cls_id`: Class ID.
  - `label`: Class name.
  - `score`: Prediction score.
  - `coordinate`: Coordinates of the bounding box, in the format <code>[xmin, ymin, xmax, ymax]</code>.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/BluebirdStory/PaddleX_doc_images/main/images/modules/mainbody_detection/general_object_detection_002_res.png"/>

**Note:** Due to network issues, the above URL may not be accessible. If you need to access this link, please check the validity of the URL and try again. If the problem persists, it may be related to the link itself or the network connection.

Related methods, parameters, and explanations are as follows:

* `create_model` instantiates a main body detection model (here, `PP-ShiTuV2_det` is used as an example), and the specific explanations are as follows:
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
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence objects</td>
<td><code>float/None/dict</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.
* `threshold` is the threshold for filtering low-confidence objects. The default is `None`, which means using the settings from the lower priority. The priority of parameter settings from highest to lowest is: `predict parameter > create_model initialization > yaml configuration file`. Currently, two types of threshold settings are supported:
  * `float`, using the same threshold for all classes.
  * `dict`, where the key is the class ID and the value is the threshold, allowing different thresholds for different classes. Since main body detection is a single-class detection, this setting is not required.

* The `predict()` method of the main body detection model is called for inference prediction. The `predict()` method has parameters `input`, `batch_size`, and `threshold`, which are explained as follows:

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
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png">Example</a></li>
<li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, elements of the list must be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
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
<td>Threshold for filtering low-confidence objects</td>
<td><code>float</code>/<code>dict</code>/<code>None</code></td>
<td>
<ul>
<li><b>None</b>, indicating the use of settings from the lower priority. The priority of parameter settings from highest to lowest is: <code>predict parameter > create_model initialization > yaml configuration file</code></li>
<li><b>float</b>, such as 0.5, indicating the use of <code>0.5</code> as the threshold for filtering low-confidence objects during inference</li>
<li><b>dict</b>, such as <code>{0: 0.5, 1: 0.35}</code>, indicating the use of 0.5 as the threshold for class 0 and 0.35 for class 1 during inference. Since main body detection is a single-class detection, this setting is not required.</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* The prediction results are processed, and the prediction result for each sample is of type `dict`. It supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3">Print the results to the terminal</td>
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
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
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
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it supports obtaining the visualization image with results and the prediction results through attributes, as follows:

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
<td rowspan="1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference APIs, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better mainbody detection models. Before developing mainbody detection models with PaddleX, ensure you have installed the PaddleDetection plugin for PaddleX. The installation process can be found in [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides a data validation function for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following commands:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/mainbody_det_examples.tar -P ./dataset
tar -xf ./dataset/mainbody_det_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:

```bash
python main.py -c paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
```
After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Details of validation results (click to expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 1,
    "train_samples": 701,
    "train_sample_paths": [
      "check_dataset/demo_img/road839.png",
      "check_dataset/demo_img/road363.png",
      "check_dataset/demo_img/road148.png"
    ],
    "val_samples": 176,
    "val_sample_paths": [
      "check_dataset/demo_img/road218.png",
      "check_dataset/demo_img/road681.png",
      "check_dataset/demo_img/road138.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  &quot;dataset_path&quot;: &quot;mainbody_det_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;COCODetDataset&quot;
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being <code>True</code> indicates that the dataset format meets the requirements. The explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>：The number of classes in this dataset is 1.</li>
<li><code>attributes.train_samples</code>：The number of samples in the training set of this dataset is 701.</li>
<li><code>attributes.val_samples</code>：The number of samples in the validation set of this dataset is 176.</li>
<li><code>attributes.train_sample_paths</code>：A list of relative paths to the visualized images of samples in the training set of this dataset.</li>
<li><code>attributes.val_sample_paths</code>： A list of relative paths to the visualized images of samples in the validation set of this dataset.</li>
</ul>
<p>The dataset validation also analyzes the distribution of sample counts across all classes in the dataset and generates a histogram (histogram.png) to visualize this distribution.</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/subj_det/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Mainbody detection does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Dataset splitting parameters can be set by modifying the <code>CheckDataset</code> section in the configuration file. Some example parameters in the configuration file are explained below:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in their original paths.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mainbody_det_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of `PP-ShiTuV2_det` as an example:

```bash
python main.py -c paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
```
The steps required are:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_det.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`
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
python main.py -c paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PP-ShiTuV2_det.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png) to your local machine.
```bash
python main.py -c paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_object_detection_002.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PP-ShiTuV2_det.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1. <b>Pipeline Integration</b>

The main body detection module can be integrated into PaddleX pipelines such as <b>General Object Detection</b> (comming soon). Simply replace the model path to update the main body detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your trained model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the main body detection module. You can refer to the Python example code in [Quick Integration](#quick), simply replace the model with the path to your trained model.
