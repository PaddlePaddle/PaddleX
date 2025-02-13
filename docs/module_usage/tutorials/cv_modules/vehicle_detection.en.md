---
comments: true
---

# Vehicle Detection Module Development Tutorial

## I. Overview
Vehicle detection is a subtask of object detection, specifically referring to the use of computer vision technology to determine the presence of vehicles in images or videos and provide specific location information for each vehicle (such as the coordinates of the bounding box). This information is of great significance for various fields such as intelligent transportation systems, autonomous driving, and video surveillance.

## II. Supported Model List


<table>
<tr>
<th>Model</th>
<th>mAP 0.5:0.95</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE-S_vehicle</td>
<td>61.3</td>
<td>9.79 / 3.48</td>
<td>54.14 / 46.69</td>
<td>28.79</td>
<td rowspan="2">Vehicle detection model based on PP-YOLOE</td>
</tr>
<tr>
<td>PP-YOLOE-L_vehicle</td>
<td>63.9</td>
<td>32.84 / 9.03</td>
<td>176.60 / 176.60</td>
<td>196.02</td>
</tr>
<b>Note: The evaluation set for the above accuracy metrics is PPVehicle dataset mAP(0.5:0.95). GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>


## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, you can complete the inference of the vehicle detection module with just a few lines of code. You can switch models under this module freely, and you can also integrate the model inference of the vehicle detection module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_detection.jpg) to your local machine.

```python
from paddlex import create_model
model_name = "PP-YOLOE-S_vehicle"
model = create_model(model_name)
output = model.predict("vehicle_detection.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```bash
{'res': "{'input_path': 'vehicle_detection.jpg', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'vehicle', 'score': 0.9574093222618103, 'coordinate': [0.10725308, 323.01917, 272.72037, 472.75375]}, {'cls_id': 0, 'label': 'vehicle', 'score': 0.9449281096458435, 'coordinate': [270.3387, 310.36923, 489.8854, 398.07562]}, {'cls_id': 0, 'label': 'vehicle', 'score': 0.939127504825592, 'coordinate': [896.4249, 292.2338, 1051.9075, 370.41345]}, {'cls_id': 0, 'label': 'vehicle', 'score': 0.9388730525970459, 'coordinate': [1057.6327, 274.0139, 1639.8386, 535.54926]}, {'cls_id': 0, 'label': 'vehicle', 'score': 0.9239683747291565, 'coordinate': [482.28885, 307.33447, 574.6905, 357.82965]}, ... ]}"}
```

The meanings of the runtime parameters are as follows:
- `input_path`: Indicates the path of the input image to be predicted.
- `page_index`: If the input is a PDF file, it represents the current page number of the PDF; otherwise, it is `None`.
- `boxes`: Information of each predicted object.
  - `cls_id`: Class ID.
  - `label`: Class name.
  - `score`: Prediction score.
  - `coordinate`: Coordinates of the predicted bounding box, in the format <code>[xmin, ymin, xmax, ymax]</code>.

The visualization image is as follows:

<img alt="Visualization Image" src="https://raw.githubusercontent.com/BluebirdStory/PaddleX_doc_images/main/images/modules/vehicle_detection/vehicle_detection_res.jpg"/>

Related methods, parameters, and explanations are as follows:

* The `create_model` method instantiates a vehicle detection model (here using `PP-YOLOE-S_vehicle` as an example), with specific explanations as follows:
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
<td>The name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>The storage path of the model</td>
<td><code>str</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>The threshold for filtering low-score objects</td>
<td><code>float/None/dict</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the built-in model parameters of PaddleX are used by default. If `model_dir` is specified, the user-defined model is used.

* The `threshold` is the threshold for filtering low-score objects. The default value is `None`, indicating that the settings from the lower priority are used. The priority order for parameter settings is: `predict parameter > create_model initialization > yaml configuration file`. Currently, two types of threshold settings are supported:
  * `float`: Use the same threshold for all classes.
  * `dict`: The key is the class ID, and the value is the threshold. Different thresholds can be set for different classes. For vehicle detection, which is a single-class detection task, this setting is not required.

* The `predict()` method of the vehicle detection model is called for inference and prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `threshold`, with specific explanations as follows:

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
<td>Data to be predicted, supports multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL Link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png">Example</a></li>
<li><b>Local Directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, elements of the list should be data of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<td>Threshold for filtering low-score objects</td>
<td><code>float</code>/<code>dict</code>/<code>None</code></td>
<td>
<ul>
<li><b>None</b>, indicating the settings from the lower priority are used. The priority order for parameter settings is: <code>predict parameter &gt; create_model initialization &gt; yaml configuration file</code></li>
<li><b>float</b>, such as 0.5, indicating the threshold of 0.5 is used for filtering low-score objects during inference</li>
<li><b>dict</b>, such as <code>{0: 0.5, 1: 0.35}</code>, indicating a threshold of 0.5 for class 0 and 0.35 for class 1 during inference. Vehicle detection is a single-class detection task and does not require this setting.</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* The prediction results are processed as `dict` type for each sample, and support operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
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
<td>Whether to format the output content with <code>JSON</code> indentation</td>
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
<td rowspan="3">Save the result as a file in <code>json</code> format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
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
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining the visualization image with results and the prediction results through attributes, as follows:

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
<td rowspan="1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better vehicle detection models. Before using PaddleX to develop vehicle detection models, please ensure that you have installed the PaddleDetection plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides a data validation function for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following commands:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/vehicle_coco_examples.tar -P ./dataset
tar -xf ./dataset/vehicle_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:

```bash
python main.py -c paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
```
After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Details of validation results (click to expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 4,
    "train_samples": 500,
    "train_sample_paths": [
      "check_dataset/demo_img/MVI_20011__img00001.jpg",
      "check_dataset/demo_img/MVI_20011__img00005.jpg",
      "check_dataset/demo_img/MVI_20011__img00009.jpg"
    ],
    "val_samples": 100,
    "val_sample_paths": [
      "check_dataset/demo_img/MVI_20032__img00401.jpg",
      "check_dataset/demo_img/MVI_20032__img00405.jpg",
      "check_dataset/demo_img/MVI_20032__img00409.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "vehicle_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being <code>True</code> indicates that the dataset format meets the requirements. The explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>：The number of classes in this dataset is 4.</li>
<li><code>attributes.train_samples</code>：The number of samples in the training set of this dataset is 500.</li>
<li><code>attributes.val_samples</code>：The number of samples in the validation set of this dataset is 100.</li>
<li><code>attributes.train_sample_paths</code>：A list of relative paths to the visualized images of samples in the training set of this dataset.</li>
<li><code>attributes.val_sample_paths</code>： A list of relative paths to the visualized images of samples in the validation set of this dataset.</li>
</ul>
<p>The dataset validation also analyzes the distribution of sample counts across all classes in the dataset and generates a histogram (histogram.png) to visualize this distribution. </p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/vehicle_det/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Vehicle detection does not support data format conversion.</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in their original paths.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of `PP-YOLOE-S_vehicle` as an example:

```bash
python main.py -c paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
```
The steps required are:

* Specify the `.yaml` configuration file path for the model (here it is `PP-YOLOE-S_vehicle.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
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
python main.py -c paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/vehicle_coco_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PP-YOLOE-S_vehicle.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to[PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. <b>Pipeline Integration</b>

The object detection module can be integrated into the [General Object Detection Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/object_detection.en.md) of PaddleX. Simply replace the model path to update the object detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your trained model.

2. <b>Module Integration</b>

The weights you produced can be directly integrated into the object detection module. You can refer to the Python example code in [Quick Integration](#三快速集成), simply replace the model with the path to your trained model.

* To perform inference predictions through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_detection.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="vehicle_detection.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PP-YOLOE-S_vehicle.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The weights you produced can be directly integrated into the vehicle detection module. You can refer to the Python example code in [Quick Integration](#iii-quick-integration), simply replace the model with the path to your trained model.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).
