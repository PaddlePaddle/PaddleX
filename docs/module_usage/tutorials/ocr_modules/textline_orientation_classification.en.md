---
comments: true
---

# Tutorial for Text Line Orientation Classification Module

## I. Overview
The text line orientation classification module primarily distinguishes the orientation of text lines and corrects them using post-processing. In processes such as document scanning and license/certificate photography, to capture clearer images, the capture device may be rotated, resulting in text lines in various orientations. Standard OCR pipelines cannot handle such data well. By utilizing image classification technology, the orientation of text lines can be predetermined and adjusted, thereby enhancing the accuracy of OCR processing.

## II. Supported Model List

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Accuracy (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Trained Model</a></td>
<td>95.54</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
<td>Text line classification model based on PP-LCNet_x0_25, with two classes: 0 degrees and 180 degrees</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on a self-built dataset covering multiple scenarios such as documents and licenses, containing 1000 images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## III. Quick Integration

> ❗ Before quick integration, please install the PaddleX wheel package first. For details, please refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After completing the installation of the wheel package, you can perform inference for the text line orientation classification module with just a few lines of code. You can switch models under this module at will, and you can also integrate the model inference of the text line orientation classification module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg) to your local machine. If the download link is not working, please check the validity of the URL and try again.

```bash
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x0_25_textline_ori")
output = model.predict("textline_rot180_demo.jpg",  batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': 'test_imgs/textline_rot180_demo.jpg', 'class_ids': [1], 'scores': [1.0], 'label_names': ['180_degree']}}
```

The meanings of the running results parameters are as follows:

- `input_path`：Indicates the path of the input image.
- `class_ids`：Indicates the class ID of the prediction result.
- `scores`：Indicates the confidence score of the prediction result.
- `label_names`：Indicates the class name of the prediction result.
The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/image_classification/general_image_classification_001_res.jpg">

The explanations for the methods, parameters, etc., are as follows:

* `create_model` instantiates a text recognition model (here, `PP-LCNet_x0_25_textline_ori` is used as an example), and the specific explanations are as follows:

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
<td><code>PP-LCNet_x0_25_textline_ori</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the text recognition model is called for inference prediction. The `predict()` method has parameters `input` and `batch_size`, which are explained as follows:

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
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg">Example</a></li>
  <li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, the elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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

For more information on using the PaddleX single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you aim for higher accuracy with existing models, you can leverage PaddleX's custom development capabilities to develop better text line orientation classification models. Before developing a text line orientation classification model with PaddleX, ensure that you have installed PaddleX's classification-related model training capabilities. The installation process can be found in the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, allowing you to complete subsequent development based on the official demo data. If you wish to use a private dataset for subsequent model training, refer to the [PaddleX Image Classification Task Module Data Preparation Tutorial](../../../data_annotations/cv_modules/image_classification.en.md).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following command:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/textline_orientation_example_data.tar -P ./dataset
tar -xf ./dataset/textline_orientation_example_data.tar -C ./dataset/
```
#### 4.1.2 Data Validation
You can complete data validation with a single command:

```bash
python main.py -c paddlex/configs/modules/textline_orientation/PP-LCNet_x0_25_textline_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/textline_orientation_example_data
```
After executing the above command, PaddleX will validate the dataset and collect basic information about it. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory under the current directory, including visualized sample images and sample distribution histograms.

<details><summary>👉 <b>Details of Verification Results (Click to Expand)</b></summary>

<p>The specific content of the verification result file is as follows:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "../../dataset/textline_orientation_example_data/label.txt",
    "num_classes": 2,
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/ILSVRC2012_val_00019234_4284.jpg",
      "check_dataset/demo_img/lsvt_train_images_4655.jpg",
      "check_dataset/demo_img/lsvt_train_images_60562.jpg",
      "check_dataset/demo_img/lsvt_train_images_14013.jpg",
      "check_dataset/demo_img/ILSVRC2012_val_00011156_12950.jpg",
      "check_dataset/demo_img/ILSVRC2012_val_00016578_10192.jpg",
      "check_dataset/demo_img/26920921_2341381071.jpg",
      "check_dataset/demo_img/31979250_3394569384.jpg",
      "check_dataset/demo_img/25959328_518853598.jpg",
      "check_dataset/demo_img/ILSVRC2012_val_00018420_14077.jpg"
    ],
    "val_samples": 200,
    "val_sample_paths": [
      "check_dataset/demo_img/lsvt_train_images_79109.jpg",
      "check_dataset/demo_img/lsvt_train_images_131133.jpg",
      "check_dataset/demo_img/mtwi_train_images_65423.jpg",
      "check_dataset/demo_img/lsvt_train_images_120718.jpg",
      "check_dataset/demo_img/mtwi_train_images_58098.jpg",
      "check_dataset/demo_img/rctw_train_images_25817.jpg",
      "check_dataset/demo_img/lsvt_val_images_6336.jpg",
      "check_dataset/demo_img/lsvt_train_images_71775.jpg",
      "check_dataset/demo_img/mtwi_train_images_78064.jpg",
      "check_dataset/demo_img/mtwi_train_images_52578.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/textline_orientation_example_data",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
</code></pre>
<p>In the above verification results, check_pass being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 2;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 1000;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 200;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of the training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of the validation samples in this dataset;</li>
</ul>
<p>The dataset verification also analyzes the distribution of sample numbers across all classes in the dataset and generates a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/textline_ori_classification/01.png"></p></details>

### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format and re-split the training/validation ratio of the dataset by **modifying the configuration file** or **adding hyperparameters**.

<details><summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Text line orientation classification temporarily does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When set to <code>True</code>, dataset splitting is performed, with a default of <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, you need to set the percentage of the training set, which is any integer between 0 and 100, and must sum to 100 with the value of <code>val_percent</code>;</li>
</ul>
<p>For example, if you want to re-split the dataset with 90% for the training set and 10% for the validation set, you need to modify the configuration file as follows:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/textline_orientation/PP-LCNet_x0_25_textline_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/textline_orientation_example_data
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/textline_orientation/PP-LCNet_x0_25_textline_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/textline_orientation_example_data \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command. Here, the training of the text line orientation classification model (PP-LCNet_x1_0_textline_ori) is taken as an example:

```bash
python main.py -c paddlex/configs/modules/textline_orientation/PP-LCNet_x0_25_textline_ori.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/textline_orientation_example_data
```
The following steps are required:

* Specify the path to the `.yaml` configuration file for the model (here it is `PP-LCNet_x0_25_textline_ori.yaml`. When training other models, you need to specify the corresponding configuration file. The correspondence between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file or by appending parameters in the command line. For example, to specify the first two GPUs for training: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file description for the corresponding task module of the model [PaddleX Common Model Configuration Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves the model weights to the default directory <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including the following:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics, loss, etc., during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configurations for this training;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

### **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weights on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/textline_orientation/PP-LCNet_x0_25_textline_ori.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/textline_orientation_example_data
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x0_25_textline_ori.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 **More Details (Click to Expand)**</summary>

<p>When evaluating the model, you need to specify the path to the model weights file. Each configuration file has a default weight save path built in. If you need to change it, you can set it by appending a command-line parameter, such as <code>-o Evaluate.weight_path="./output/best_model/best_model.pdparams"</code>.</p>
<p>After completing the model evaluation, the following outputs are typically generated:</p>
<p>Upon completion of the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results. Specifically, it records whether the evaluation task was completed normally and the model's evaluation metrics, including Top-1 Accuracy.</p></details>

### **4.4 Model Inference and Model Integration**
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
Performing inference predictions through the command line requires only the following single command. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/textline_rot180_demo.jpg) locally.

```bash
python main.py -c paddlex/configs/modules/textline_orientation/PP-LCNet_x0_25_textline_ori.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="textline_rot180_demo.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x0_25_textline_ori.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the path to the model weights: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the path to the input data: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. **Pipeline Integration**

The text line orientation classification module can be integrated into the [Document Scene Information Extraction v3 Pipeline (PP-ChatOCRv3-doc)](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.en.md). Simply replace the model path to update the text line orientation classification module.

2. **Module Integration**

The weights you produce can be directly integrated into the text line orientation classification module. You can refer to the Python example code in [Quick Integration](##Quick-Integration) and only need to replace the model with the path to your trained model.
