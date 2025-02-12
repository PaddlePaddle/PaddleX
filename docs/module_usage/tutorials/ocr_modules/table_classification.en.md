---
comments: true
---

# Table Classification Module Usage Tutorial

## I. Overview
The table classification module is a key component of a computer vision system, responsible for classifying input table images. The performance of this module directly affects the accuracy and efficiency of the entire table recognition process. The table classification module typically receives table images as input and then, through deep learning algorithms, classifies them into predefined categories based on the characteristics and content of the images, such as wired tables and wireless tables. The classification results of the table classification module are provided as output for use in table recognition-related production lines.

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training Model</a></td>
<td>--</td>
<td>--</td>
<td>--</td>
<td>6.6M</td>
</tr>
</table>

<p><b>Note: The above accuracy metrics are measured from the internal table classification dataset built by PaddleX. All models' GPU inference time is based on an NVIDIA Tesla T4 machine, with a precision type of FP32. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and a precision type of FP32.</b></p></details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package first. For details, please refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can complete the inference of the table classification module with just a few lines of code. You can switch between models under this module at will, and you can also integrate the model inference of the table classification module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your local machine. 

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0_table_cls")
output = model.predict("table_recognition.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```

After running the code, the result obtained is:

```json
{'res': {'input_path': 'table_recognition.jpg', 'class_ids': array([0, 1], dtype=int32), 'scores': array([0.84421, 0.15579], dtype=float32), 'label_names': ['wired_table', 'wireless_table']}}
```

The meanings of the parameters in the running results are as follows:
- `input_path`: Indicates the path of the input image.
- `class_ids`: Indicates the class ID of the prediction result.
- `scores`: Indicates the confidence of the prediction result.
- `label_names`: Indicates the class name of the prediction result.

The descriptions of the related methods and parameters are as follows:

* `create_model` instantiates a table classification model (here we use `PP-LCNet_x1_0_table_cls` as an example), and the specific descriptions are as follows:
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
<td><code>model_name</code></td>
<td>Model name</td>
<td><code>str</code></td>
<td>No</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path</td>
<td><code>str</code></td>
<td>No</td>
<td><code>None</code></td>
</tr>
</table>

* The `model_name` must be specified. After specifying the `model_name`, the default model parameters in PaddleX will be used. On this basis, if `model_dir` is specified, the user-defined model will be used.

* Call the `predict()` method of the table classification model to perform inference prediction. The `predict()` method has parameters `input` and `batch_size`, and the specific descriptions are as follows:

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
  <li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">Example</a></li>
  <li><b>Local directory</b>, this directory should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. It is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. It is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file will be named consistently with the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. It is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. It is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
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
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction result in <code>json</code> format</td>
</tr>
</table>

For more information on the usage of PaddleX's single-model inference API, please refer to [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Secondary Development
If you aim to improve the accuracy of existing models, you can leverage PaddleX's secondary development capabilities to develop a better table classification model. Before using PaddleX to develop a table classification model, please ensure that you have installed the table classification part of PaddleX according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before training the model, you need to prepare the dataset for the corresponding task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. In addition, PaddleX provides a demo dataset for each module, and you can complete subsequent development based on the official demo data. If you want to use your private dataset for model training, please refer to the [PaddleX Image Classification Task Module Data Annotation Guide](../../../data_annotations/cv_modules/image_classification.en.md).

#### 4.1.1 Downloading Demo Data
You can use the following command to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_cls_examples.tar -P ./dataset
tar -xf ./dataset/table_cls_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
Data validation can be completed with a single line of command:

```bash
python main.py -c paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/table_cls_examples
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. If the command runs successfully, it will print the message `Check dataset passed !` in the log. The verification result file is saved at `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory under the current directory. This output directory includes visualized example sample images and sample distribution histograms.

<details><summary>👉 <b>Verification Result Details (Click to Expand)</b></summary>

```json
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "..\/..\/..\/docs_for_rc\/test_for_doc\/table_cls_examples\/label.txt",
    "num_classes": 2,
    "train_samples": 410,
    "train_sample_paths": [
      "check_dataset\/demo_img\/img_14707_0.png",
      "check_dataset\/demo_img\/img_14346_1.png",
      "check_dataset\/demo_img\/img_14707_3.png",
      "check_dataset\/demo_img\/img_12881_4.png",
      "check_dataset\/demo_img\/img_1676_4.png",
      "check_dataset\/demo_img\/img_14909_3.png",
      "check_dataset\/demo_img\/img_3530_4.png",
      "check_dataset\/demo_img\/img_5471_4.png",
      "check_dataset\/demo_img\/img_8396_4.png",
      "check_dataset\/demo_img\/img_13019_2.png"
    ],
    "val_samples": 102,
    "val_sample_paths": [
      "check_dataset\/demo_img\/img_4345_3.png",
      "check_dataset\/demo_img\/img_15063_0.png",
      "check_dataset\/demo_img\/img_747_3.png",
      "check_dataset\/demo_img\/img_5535_2.png",
      "check_dataset\/demo_img\/img_15250_2.png",
      "check_dataset\/demo_img\/img_4791_4.png",
      "check_dataset\/demo_img\/img_2562_2.png",
      "check_dataset\/demo_img\/img_15248_2.png",
      "check_dataset\/demo_img\/img_4178_3.png",
      "check_dataset\/demo_img\/img_11090_0.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "table_cls_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
```

<p>In the above verification results, <code>check_pass</code> being <code>True</code> indicates that the dataset format meets the requirements. The explanations for the other metrics are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 2;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 410;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 102;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths for the visualization images of the training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths for the visualization images of the validation samples in this dataset;</li>
</ul>
<p>In addition, the dataset verification has analyzed the distribution of sample counts for all classes in the dataset and generated a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/table_classification/01.png"></p></details></url> 

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After you complete the data verification, you can convert the dataset format by <b>modifying the configuration file</b> or <b>adding hyperparameters</b>. You can also re-split the training/validation ratio of the dataset.

<details><summary>👉 <b>Details of Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Table classification does not support data conversion at the moment.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example explanations of the parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When set to <code>True</code>, the dataset will be re-split. The default value is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, you need to set the percentage of the training set. It should be an integer between 0 and 100, and it must sum up to 100 with the value of <code>val_percent</code>;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/table_cls_examples
</code></pre>
<p>After the dataset splitting is executed, the original annotation file will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/table_cls_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Training a model can be done with a single command. For example, to train the table classification model PP-LCNet_x1_0_table_cls:

```bash
python main.py -c paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/table_cls_examples
```

* Specify the `.yaml` configuration file path for the model (here it is `PP-LCNet_x1_0_table_cls.yaml`. When training other models, the corresponding configuration file needs to be specified. The correspondence between models and configurations can be found in [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file description of the corresponding model task module [PaddleX General Model Configuration Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX will automatically save the model weight files, with the default directory being <code>output</code>. If you need to specify a different save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts away the concepts of dynamic graph weights and static graph weights for you. During model training, both dynamic and static graph weights will be generated. By default, static graph weights are used for model inference.</li>
<li>
<p>After model training is completed, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including the following:</p>
</li>
<li>
<p><code>train_result.json</code>: The training result record file, which logs whether the training task was completed successfully, as well as the metrics of the generated weights and related file paths;</p>
</li>
<li><code>train.log</code>: The training log file, which records changes in model metrics and loss during the training process;</li>
<li><code>config.yaml</code>: The training configuration file, which logs the hyperparameter settings for this training session;</li>
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, and static graph network structure, etc.</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. With PaddleX, model evaluation can be completed with a single command:

```bash
python main.py -c  paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/table_cls_examples
```

Similar to model training, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_table_cls.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the validation dataset path: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the path of the model weight file. Each configuration file has a default weight save path built in. If you need to change it, you can simply set it by appending command-line parameters, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed normally and the model's evaluation metrics, including val.top1 and val.top5.</p></details>

### <b>4.4 Model Inference and Model Integration</b>

After completing the training and evaluation of the model, you can use the trained model weights for inference prediction or integrate them into Python.

#### 4.4.1 Model Inference
Inference prediction can be performed via the command line with just one command. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your local machine. Note that due to network issues, the link may not be accessible. If you encounter any problems, please check the validity of the link and try again.

```bash
python main.py -c paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="table_recognition.jpg"
```

Similar to model training and evaluation, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_table_cls.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`

Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX production line or directly integrated into your own project.

1.<b>Production Line Integration</b>

The table classification module can be integrated into the PaddleX production line such as [General Table Classification Pipeline v2](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.en.md). You just need to replace the model path to update the table classification module in the related production line. In production line integration, you can deploy the model you obtained using high-performance deployment and service-oriented deployment.

2.<b>Module Integration</b>

The weights you generate can be directly integrated into the table classification module. You can refer to the Python example code in [Quick Integration](#Three-Quick-Integration). Just replace the model with the path of the model you have trained.