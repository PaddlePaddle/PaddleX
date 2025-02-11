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
<td>2.32 / 2.32</td>
<td>3.22 / 1.26</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_vehicle_attribute is a lightweight vehicle attribute recognition model based on PP-LCNet.</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are mA on the VeRi dataset. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>


## <span id="lable">III. Quick Integration</span>

&gt; ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the vehicle attribute recognition module. You can easily switch models under this module, and you can also integrate the model inference of the vehicle attribute recognition module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_007.jpg) to your local machine.

```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0_vehicle_attribute")
output = model.predict("vehicle_attribute_007.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
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
  "dataset_path": "./dataset/vehicle_attribute_examples",
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

The vehicle attribute recognition module can be integrated into the [General Image Multi-label Classification Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.en.md) of PaddleX. Simply replace the model path to update the vehicle attribute recognition module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the vehicle attribute recognition module. Refer to the Python example code in  <a href="#lable">Quick Integration</a>  and simply replace the model with the path to your trained model.
