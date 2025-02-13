---
comments: true
---

# Vehicle Attribute Recognition Module Development Tutorial

## I. Overview
Vehicle attribute recognition is a crucial component in computer vision systems. Its primary task is to locate and label specific attributes of vehicles in images or videos, such as vehicle type, color, license plate number, etc. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The vehicle attribute recognition module typically outputs bounding boxes (Bounding Boxes) containing vehicle attribute information, which are then passed as input to other modules (e.g., vehicle tracking, vehicle re-identification) for subsequent processing.

## II. Supported Model List


<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mA (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_vehicle_attribute</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_vehicle_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_vehicle_attribute_pretrained.pdparams">Trained Model</a></td>
<td>91.7</td>
<td>2.32 / 0.52</td>
<td>3.22 / 1.26</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_vehicle_attribute is a lightweight vehicle attribute recognition model based on PP-LCNet.</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are mA on the VeRi dataset. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>


## <span id="lable">III. Quick Integration</span>

> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the vehicle attribute recognition module. You can easily switch models under this module, and you can also integrate the model inference of the vehicle attribute recognition module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_007.jpg) to your local machine.

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0_vehicle_attribute")
output = model.predict("vehicle_attribute_007.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the obtained result is:

```bash
{'res': {'input_path': 'vehicle_attribute_007.jpg', 'page_index': None, 'class_ids': array([ 0, 13]), 'scores': array([0.98929, 0.97349]), 'label_names': ['yellow(黄色)', 'hatchback(掀背车)']}}
```

The meanings of the parameters in the running result are as follows:
- `input_path`: Indicates the path of the input multi-category image to be predicted.
- `page_index`: If the input is a PDF file, it indicates which page of the PDF is currently being processed; otherwise, it is `None`.
- `class_ids`: Indicates the predicted label IDs of the vehicle attribute images.
- `scores`: Indicates the confidence scores of the predicted labels of the vehicle attribute images.
- `label_names`: Indicates the names of the predicted labels of the vehicle attribute images.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/vehicle_attri/vehicle_attribute_007_res.jpg" alt="Vehicle Attribute Result">

Relevant methods, parameters, and explanations are as follows:

* `create_model` instantiates the vehicle attribute recognition model (here, `PP-LCNet_x1_0_vehicle_attribute` is used as an example). The specific explanations are as follows:
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
<td>The name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td><code>PP-LCNet_x1_0_vehicle_attribute</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>The storage path of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>The threshold for vehicle attribute recognition</td>
<td><code>float/list/dict</code></td>
<td><li><b>float variable</b>, any floating-point number between [0-1]: <code>0.5</code></li>
<li><b>list variable</b>, a list composed of multiple floating-point numbers between [0-1]: <code>[0.5,0.5,...]</code></li>
<li><b>dict variable</b>, specifying different thresholds for different categories, where "default" is a required key: <code>{"default":0.5,1:0.1,...}</code></li>
</td>
<td>0.5</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, PaddleX's built-in model parameters are used by default. If `model_dir` is specified, the user-defined model is used.

* The `threshold` parameter is used to set the threshold for multi-label classification, with a default value of 0.7. When set as a float, it means all categories use this threshold; when set as a list, different categories use different thresholds, and the list length must match the number of categories; when set as a dictionary, "default" is a required key, indicating the default threshold for all categories, while other categories use their respective thresholds. For example: <code>{"default":0.5,1:0.1}</code>.

* The `predict()` method of the multi-label classification model is called for inference prediction. The parameters of the `predict()` method include `input`, `batch_size`, and `threshold`, with specific explanations as follows:

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
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png">Example</a></li>
  <li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, elements of the list should be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<td>Threshold for vehicle attribute recognition</td>
<td><code>float/list/dict</code></td>
<td><li><b>float variable</b>, any floating-point number between [0-1]: <code>0.5</code></li>
<li><b>list variable</b>, a list of multiple floating-point numbers between [0-1]: <code>[0.5,0.5,...]</code></li>
<li><b>dict variable</b>, specifying different thresholds for different categories, where <code>"default"</code> is a required key: <code>{"default":0.5,1:0.1,...}</code></li>
</td>
<td>0.5</td>
</tr>
</table>

* The prediction results are processed, with each sample's prediction result being a corresponding Result object, which supports operations such as printing, saving as an image, and saving as a <code>json</code> file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Description</th>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a <code>json</code> file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving; if it is a directory, the saved file will be named consistently with the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving; if it is a directory, the saved file will be named consistently with the input file type</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining visualized images with results and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td>Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "1"><code>img</code></td>
<td>Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference API, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

<b>Note</b>: In the `output`, values indexed from 0-9 represent color attributes, corresponding to the following colors respectively: yellow, orange, green, gray, red, blue, white, golden, brown, black. Indices 10-18 represent vehicle type attributes, corresponding to the following vehicle types: sedan, suv, van, hatchback, mpv, pickup, bus, truck, estate.


## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better vehicle attribute recognition models. Before using PaddleX to develop vehicle attribute recognition models, ensure you have installed the classification-related model training plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Multi-Label Classification Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/ml_classification.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/vehicle_attribute_examples.tar -P ./dataset
tar -xf ./dataset/vehicle_attribute_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_attribute_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "../../dataset/vehicle_attribute_examples/label.txt",
    "num_classes": 19,
    "train_samples": 1200,
    "train_sample_paths": [
      "check_dataset/demo_img/0018_c017_00033140_0.jpg",
      "check_dataset/demo_img/0010_c019_00034275_0.jpg",
      "check_dataset/demo_img/0015_c019_00068660_0.jpg",
      "check_dataset/demo_img/0016_c017_00049590_1.jpg",
      "check_dataset/demo_img/0018_c016_00052280_0.jpg",
      "check_dataset/demo_img/0023_c001_00006995_0.jpg",
      "check_dataset/demo_img/0022_c004_00065910_0.jpg",
      "check_dataset/demo_img/0007_c019_00048655_1.jpg",
      "check_dataset/demo_img/0022_c007_00072970_0.jpg",
      "check_dataset/demo_img/0022_c008_00065785_0.jpg"
    ],
    "val_samples": 300,
    "val_sample_paths": [
      "check_dataset/demo_img/0025_c003_00054095_0.jpg",
      "check_dataset/demo_img/0023_c013_00006350_1.jpg",
      "check_dataset/demo_img/0024_c003_00046320_0.jpg",
      "check_dataset/demo_img/0025_c005_00054795_2.jpg",
      "check_dataset/demo_img/0024_c012_00041770_0.jpg",
      "check_dataset/demo_img/0024_c007_00060845_1.jpg",
      "check_dataset/demo_img/0023_c017_00013150_0.jpg",
      "check_dataset/demo_img/0024_c014_00040410_0.jpg",
      "check_dataset/demo_img/0025_c002_00050685_1.jpg",
      "check_dataset/demo_img/0025_c005_00032645_0.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "vehicle_attribute_examples",
  "show_type": "image",
  "dataset_type": "MLClsDataset"
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 19;</li>
<li><code>attributes.train_samples</code>: The number of samples in the training set of this dataset is 1200;</li>
<li><code>attributes.val_samples</code>: The number of samples in the validation set of this dataset is 300;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of samples in the training set of this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of samples in the validation set of this dataset;</li>
</ul>
<p>Additionally, the dataset verification also analyzes the distribution of the length and width of all images in the dataset and plots a histogram (histogram.png):
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/vehicle_attri/01.png"/></p></details>


#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Vehicle attribute recognition does not support dataset format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The dataset splitting parameters can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. An example of part of the configuration file is shown below:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The value should be an integer between 0 and 100, and the sum with <code>val_percent</code> should be 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with 90% training set and 10% validation set, modify the configuration file as follows:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_attribute_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/vehicle_attribute_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Training a model can be done with a single command, taking the training of the PP-LCNet vehicle attribute recognition model (`PP-LCNet_x1_0_vehicle_attribute`) as an example:

```bash
python main.py -c paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/vehicle_attribute_examples
```
The steps required are:

* Specify the path to the model's `.yaml` configuration file (here it's `PP-LCNet_x1_0_vehicle_attribute.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves the model weight files, with the default being <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
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
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/vehicle_attribute_examples
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_vehicle_attribute.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).


<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including MultiLabelMAP;</p></details>

### <b>4.4 Model Inference and Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.


#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_007.jpg) to your local machine.

```bash
python main.py -c paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="vehicle_attribute_007.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-LCNet_x1_0_vehicle_attribute.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.<b>Pipeline Integration</b>

The vehicle attribute recognition module can be integrated into the [Vehicle Attribute Recognition Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/vehicle_attribute_recognition.en.md) of PaddleX. Simply replace the model path to update the vehicle attribute recognition module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the vehicle attribute recognition module. Refer to the Python example code in  <a href="#lable">Quick Integration</a>  and simply replace the model with the path to your trained model.
