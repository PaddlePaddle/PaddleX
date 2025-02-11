---
comments: true
---

# Tutorial on Using the Human Keypoint Detection Module

## I. Overview
Human keypoint detection is an important task in the field of computer vision, aiming to identify the specific keypoint locations of the human body in images or videos. By detecting these keypoints, various applications such as pose estimation, action recognition, human-computer interaction, and animation generation can be achieved. Human keypoint detection has a wide range of applications in augmented reality, virtual reality, motion capture, and other fields.

Keypoint detection algorithms mainly include two approaches: Top-Down and Bottom-Up. The Top-Down approach typically relies on an object detection algorithm to identify the bounding boxes of the objects of interest. The input to the keypoint detection model is a cropped single object, and the output is the keypoint prediction result for that object. The model's accuracy is higher, but its speed slows down with an increasing number of objects. In contrast, the Bottom-Up method does not rely on prior object detection but directly performs keypoint detection on the entire image, then groups or connects these points to form multiple pose instances. Its speed is fixed and does not slow down with an increasing number of objects, but its accuracy is lower.

## II. Supported Model List

<table>
  <tr>
    <th>Model</th>
    <th>Approach</th>
    <th>Input Size</th>
    <th>AP(0.5:0.95)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time (ms)</th>
    <th>Model Size (M)</th>
    <th>Introduction</th>
  </tr>
  <tr>
    <td>PP-TinyPose_128x96</td>
    <td>Top-Down</td>
    <td>128x96</td>
    <td>58.4</td>
    <td></td>
    <td></td>
    <td>4.9</td>
    <td rowspan="2">PP-TinyPose is a real-time keypoint detection model optimized for mobile devices developed by the Baidu PaddlePaddle Vision Team. It can smoothly perform multi-person pose estimation tasks on mobile devices.</td>
  </tr>
  <tr>
    <td>PP-TinyPose_256x192</td>
    <td>Top-Down</td>
    <td>256x192</td>
    <td>68.3</td>
    <td></td>
    <td></td>
    <td>4.9</td>
  </tr>
</table>

**Note: The above accuracy metrics are based on the COCO dataset AP(0.5:0.95) using ground truth annotations for bounding boxes. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision, while CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package first. For details, please refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After completing the installation of the wheel package, you can perform inference for the human keypoint detection module with just a few lines of code. You can switch models under this module at will, and you can also integrate the model inference of the human keypoint detection module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_002.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "PP-TinyPose_128x96"

model = create_model(model_name)
output = model.predict("keypoint_detection_002.jpg", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

```bash
{'res': {'input_path': 'keypoint_detection_002.jpg', 'kpts': [{'keypoints': [[175.2838134765625, 56.043609619140625, 0.6522828936576843], [181.32794189453125, 49.642051696777344, 0.7338210940361023], [169.46002197265625, 50.59111022949219, 0.6837076544761658], [193.3421173095703, 51.91969680786133, 0.8676544427871704], [164.50787353515625, 55.6519889831543, 0.8232858777046204], [219.7235870361328, 90.28710174560547, 0.8812915086746216], [152.90377807617188, 95.07806396484375, 0.9093065857887268], [233.1095733642578, 149.6704864501953, 0.7706904411315918], [139.5576629638672, 144.38327026367188, 0.7555014491081238], [245.22830200195312, 202.4243927001953, 0.706590473651886], [117.83794403076172, 188.56410217285156, 0.8892115950584412], [203.29542541503906, 200.2967071533203, 0.838330864906311], [172.00791931152344, 201.1993865966797, 0.7636935710906982], [181.18797302246094, 273.0669250488281, 0.8719099164009094], [185.1750030517578, 278.4797668457031, 0.6878190040588379], [171.55068969726562, 362.42730712890625, 0.7994316816329956], [201.6941375732422, 354.5953369140625, 0.6789217591285706]], 'kpt_score': 0.7831441760063171}]}}
```

<details><summary>👉 <b>The result obtained after running is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'keypoint_detection_002.jpg', 'kpts': [{'keypoints': [[175.2838134765625, 56.043609619140625, 0.6522828936576843], [181.32794189453125, 49.642051696777344, 0.7338210940361023], [169.46002197265625, 50.59111022949219, 0.6837076544761658], [193.3421173095703, 51.91969680786133, 0.8676544427871704], [164.50787353515625, 55.6519889831543, 0.8232858777046204], [219.7235870361328, 90.28710174560547, 0.8812915086746216], [152.90377807617188, 95.07806396484375, 0.9093065857887268], [233.1095733642578, 149.6704864501953, 0.7706904411315918], [139.5576629638672, 144.38327026367188, 0.7555014491081238], [245.22830200195312, 202.4243927001953, 0.706590473651886], [117.83794403076172, 188.56410217285156, 0.8892115950584412], [203.29542541503906, 200.2967071533203, 0.838330864906311], [172.00791931152344, 201.1993865966797, 0.7636935710906982], [181.18797302246094, 273.0669250488281, 0.8719099164009094], [185.1750030517578, 278.4797668457031, 0.6878190040588379], [171.55068969726562, 362.42730712890625, 0.7994316816329956], [201.6941375732422, 354.5953369140625, 0.6789217591285706]], 'kpt_score': 0.7831441760063171}]}}
```

Parameter meanings are as follows:
- `input_path`: The path of the input image to be predicted.
- `kpts`: Information of the predicted keypoints, a list of dictionaries. Each dictionary contains the following information:
  - `keypoints`: Keypoint coordinate information, a numpy array with shape [num_keypoints, 3], where each keypoint is composed of [x, y, score], and score represents the confidence of that keypoint.
  - `kpt_score`: The overall confidence of the keypoints, i.e., the average confidence of the keypoints.

</details>

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/keypoint_det/keypoint_detection_002_res.jpg">

The explanations for the methods, parameters, etc., are as follows:

* `create_model` instantiates a human keypoint detection model (here, `PP-TinyPose_128x96` is used as an example), and the specific explanations are as follows:
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
<td><code>flip</code></td>
<td>Whether to perform flipped inference; if True, the model will infer the horizontally flipped input image and fuse the results of both inferences to increase the accuracy of keypoint predictions</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>False</code></td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the human keypoint detection model is called for inference prediction. The `predict()` method has parameters `input` and `batch_size`, which are explained as follows:

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
  <li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
  <li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>Dictionary</b>, the <code>key</code> of the dictionary must correspond to the specific task, such as <code>"img"</code> for image classification tasks. The <code>value</code> of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code></li>
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

For more information on using the PaddleX single-model inference API, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Secondary Development
If you aim to improve the accuracy of existing models, you can leverage PaddleX's secondary development capabilities to create better keypoint detection models. Before developing keypoint detection models with PaddleX, make sure to install the PaddleDetection plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before training a model, you need to prepare the dataset for the specific task module. PaddleX provides a data validation feature for each module, and **only datasets that pass the validation can be used for model training**. Additionally, PaddleX offers demo datasets for each module, which you can use to complete subsequent development based on the official demo data. If you wish to use your private dataset for model training, please refer to the [PaddleX Keypoint Detection Data Annotation Guide](../../../data_annotations/cv_modules/keypoint_detection.en.md).

#### 4.1.1 Downloading Demo Data
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/keypoint_coco_examples.tar -P ./dataset
tar -xf ./dataset/keypoint_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete the data validation:

```bash
python main.py -c paddlex/configs/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples
````

After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation result file is saved at `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset `directory under the current directory. This includes visualized sample images and sample distribution histograms.

<details>
  <summary>👉 <b>Validation Result Details (Click to expand)</b></summary>

The content of the validation result file is as follows:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 1,
    "train_samples": 500,
    "train_sample_paths": [
      "check_dataset/demo_img/000000560108.jpg",
      "check_dataset/demo_img/000000434662.jpg",
      "check_dataset/demo_img/000000540556.jpg",
      ...
    ],
    "val_samples": 100,
    "val_sample_paths": [
      "check_dataset/demo_img/000000463730.jpg",
      "check_dataset/demo_img/000000085329.jpg",
      "check_dataset/demo_img/000000459153.jpg",
      ...
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "keypoint_coco_examples",
  "show_type": "image",
  "dataset_type": "KeypointTopDownCocoDetDataset"
}
```

In the above validation results, `check_pass` being `True` indicates that the dataset format meets the requirements. Explanations for other metrics are as follows:

* `attributes.num_classes`: The dataset contains 1 class.
* `attributes.train_samples`: The training set contains 500 samples.
* `attributes.val_samples`: The validation set contains 100 samples.
* `attributes.train_sample_paths`: A list of relative paths to visualized training samples.
* `attributes.val_sample_paths`: A list of relative paths to visualized validation samples.

The data validation also analyzes the sample distribution across all classes in the dataset and generates a histogram (histogram.png):

![Histogram](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/keypoint_det/01.png)

</details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format or re-split the training/validation ratio by **modifying the configuration file** or **adding hyperparameters**.

<details>
  <summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to expand)</b></summary>

**（1）Dataset Format Conversion**

Keypoint detection does not support dataset format conversion.

**（2）Dataset Splitting**

Parameters for dataset splitting can be set by modifying the fields under `CheckDataset` in the configuration file. Example explanations for some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. Set to `True` to enable dataset splitting. Default is `False`.
    * `train_percent`: If re-splitting the dataset, set the percentage of the training set. This should be an integer between 0-100, ensuring the sum with `val_percent` equals 100.

For example, if you want to re-split the dataset with 90% for training and 10% for validation, modify the configuration file as follows:

```bash
......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
````

Then execute the following command:

```bash
python main.py -c paddlex/configs/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples
```

After the dataset splitting is executed, the original annotation file will be renamed to xxx.bak in the original path.

The above parameters can also be set via command-line arguments:

```bash
python main.py -c paddlex/configs/keypoint_detection/PP-TinyPose_128x96.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:


<pre><code class="language-bash">python main.py -c paddlex/configs/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples
</code></pre>

Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `MobileFaceNet.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details>
<summary>👉 <b>More Details (Click to Expand)</b></summary>

During model evaluation, the path to the model weights file needs to be specified. Each configuration file has a default weight save path built in. If you need to change it, you can set it by appending a command line parameter, such as `-o Evaluate.weight_path="./output/best_model/best_model/model.pdparams"`.

After completing the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results. Specifically, it records whether the evaluation task was completed normally and the model's evaluation metrics, including Accuracy.</details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be implemented through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, you only need the following command. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_002.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="keypoint_detection_002.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `MobileFaceNet.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the path to the model weights: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the path to the input data: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. <b>Pipeline Integration</b>

The human keypoint detection module can be integrated into the PaddleX pipeline for [**human keypoint detection**](../../../pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.en.md). Simply replacing the model path will update the human keypoint detection module in the relevant pipeline. In pipeline integration, you can deploy your model using high-performance deployment or service-oriented deployment.

2. <b>Module Integration</b>

The weights you produced can be directly integrated into the face feature module. You can refer to the Python example code in [Quick Integration](#III.-Quick-Integration) and only need to replace the model with the path to the model you trained.
