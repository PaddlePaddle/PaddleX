---
comments: true
---

# Pedestrian Attribute Recognition Module Development Tutorial

## I. Overview
Pedestrian attribute recognition is a crucial component in computer vision systems, responsible for locating and labeling specific attributes of pedestrians in images or videos, such as gender, age, clothing color, and type. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The pedestrian attribute recognition module typically outputs attribute information for each pedestrian, which is then passed as input to other modules (e.g., pedestrian tracking, pedestrian re-identification) for subsequent processing.

## II. Supported Model List


<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mA (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_pedestrian_attribute</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams">Trained Model</a></td>
<td>92.2</td>
<td>2.35 / 0.49</td>
<td>3.17 / 1.25</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_pedestrian_attribute is a lightweight pedestrian attribute recognition model based on PP-LCNet, covering 26 categories</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are mA on PaddleX's internal self-built dataset. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## <span id="lable">III. Quick Integration</span>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the pedestrian attribute recognition module. You can easily switch models under this module and integrate the model inference of pedestrian attribute recognition into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_006.jpg) to your local machine.

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0_pedestrian_attribute")
output = model.predict("pedestrian_attribute_006.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the obtained result is:

```bash
{'res': {'input_path': 'pedestrian_attribute_006.jpg', 'page_index': None, 'class_ids': array([10, ..., 23]), 'scores': array([1.     , ..., 0.54777]), 'label_names': ['LongCoat(长外套)', 'Age18-60(年龄在18-60岁之间)', 'Trousers(长裤)', 'Front(面朝前)']}}
```

<b>Note</b>: The index of the `class_ids` value represents the following attributes: index 0 indicates whether a hat is worn, index 1 indicates whether glasses are worn, indexes 2-7 represent the style of the upper garment, indexes 8-13 represent the style of the lower garment, index 14 indicates whether boots are worn, indexes 15-17 represent the type of bag carried, index 18 indicates whether an object is held in front, indexes 19-21 represent age, index 22 represents gender, and indexes 23-25 represent orientation. Specifically, the attributes include the following types:

```
- Gender: Male, Female
- Age: Under 18, 18-60, Over 60
- Orientation: Front, Back, Side
- Accessories: Glasses, Hat, None
- Holding Object in Front: Yes, No
- Bag: Backpack, Shoulder Bag, Handbag
- Upper Garment Style: Striped, Logo, Plaid, Patchwork
- Lower Garment Style: Striped, Patterned
- Short-sleeved Shirt: Yes, No
- Long-sleeved Shirt: Yes, No
- Long Coat: Yes, No
- Pants: Yes, No
- Shorts: Yes, No
- Skirt: Yes, No
- Boots: Yes, No
```

The meanings of the parameters in the running result are as follows:
- `input_path`: Indicates the path of the input multi-category image to be predicted.
- `page_index`: If the input is a PDF file, it indicates which page of the PDF is currently being processed; otherwise, it is `None`.
- `class_ids`: Indicates the predicted label IDs of the pedestrian attribute images.
- `scores`: Indicates the confidence scores of the predicted labels of the pedestrian attribute images.
- `label_names`: Indicates the names of the predicted labels of the pedestrian attribute images.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/ped_attri/pedestrian_attribute_006_res.jpg" alt="Pedestrian Attribute Result">

Relevant methods, parameters, and explanations are as follows:

* `create_model` instantiates the vehicle attribute recognition model (here, `PP-LCNet_x1_0_pedestrian_attribute` is used as an example). The specific explanations are as follows:
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
<td><code>PP-LCNet_x1_0_pedestrian_attribute</code></td>
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
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Optional</th>
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
<td>Threshold for pedestrian attribute recognition</td>
<td><code>float/list/dict</code></td>
<td>
<ul>
<li><b>Float variable</b>, any floating-point number between [0-1]: <code>0.5</code></li>
<li><b>List variable</b>, a list composed of multiple floating-point numbers between [0-1]: <code>[0.5,0.5,...]</code></li>
<li><b>Dict variable</b>, specifying different thresholds for different categories, where "default" is a required key: <code>{"default":0.5,1:0.1,...}</code></li>
</ul>
</td>
<td>0.5</td>
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
<td>The file path to save the result. When it is a directory, the saved file name will be consistent with the input file name</td>
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
<td>The file path to save the result. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining the visualized image with results and the prediction results through attributes, as follows:

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
<tr>
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

For more information on the usage of PaddleX single-model inference APIs, you can refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better pedestrian attribute recognition models. Before developing pedestrian attribute recognition with PaddleX, ensure you have installed the classification-related model training plugins for PaddleX.  The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](../../../installation/installation.en.md).


### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to the [PaddleX Multi-Label Classification Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/ml_classification.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/pedestrian_attribute_examples.tar -P ./dataset
tar -xf ./dataset/pedestrian_attribute_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
Run a single command to complete data validation:

```bash
python main.py -c paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "../../dataset/pedestrian_attribute_examples/label.txt",
    "num_classes": 26,
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/020907.jpg",
      "check_dataset/demo_img/004274.jpg",
      "check_dataset/demo_img/009412.jpg",
      "check_dataset/demo_img/026873.jpg",
      "check_dataset/demo_img/030560.jpg",
      "check_dataset/demo_img/022846.jpg",
      "check_dataset/demo_img/009055.jpg",
      "check_dataset/demo_img/015399.jpg",
      "check_dataset/demo_img/006435.jpg",
      "check_dataset/demo_img/055307.jpg"
    ],
    "val_samples": 500,
    "val_sample_paths": [
      "check_dataset/demo_img/080381.jpg",
      "check_dataset/demo_img/080469.jpg",
      "check_dataset/demo_img/080146.jpg",
      "check_dataset/demo_img/080003.jpg",
      "check_dataset/demo_img/080283.jpg",
      "check_dataset/demo_img/080104.jpg",
      "check_dataset/demo_img/080149.jpg",
      "check_dataset/demo_img/080313.jpg",
      "check_dataset/demo_img/080131.jpg",
      "check_dataset/demo_img/080412.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "pedestrian_attribute_examples",
  "show_type": "image",
  "dataset_type": "MLClsDataset"
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 26;</li>
<li><code>attributes.train_samples</code>: The number of samples in the training set of this dataset is 1000;</li>
<li><code>attributes.val_samples</code>: The number of samples in the validation set of this dataset is 500;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of samples in the training set of this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of samples in the validation set of this dataset;</li>
</ul>
<p>Additionally, the dataset verification also analyzes the distribution of the length and width of all images in the dataset and plots a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/ped_attri/image.png"/></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Pedestrian attribute recognition does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The dataset splitting parameters can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. An example of part of the configuration file is shown below:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>


### 4.2 Model Training
Model training can be completed with a single command. Taking the training of the PP-LCNet pedestrian attribute recognition model (PP-LCNet_x1_0_pedestrian_attribute) as an example:

```bash
python main.py -c paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
```
the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_pedestrian_attribute.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path of the training dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file parameter instructions for the corresponding task module of the model [PaddleX Common Model Configuration File Parameters](../../instructions/config_parameters_common.en.md).


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
python main.py -c paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_pedestrian_attribute.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including MultiLabelMAP;</p></details>

### <b>4.4 Model Inference and Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_006.jpg) to your local machine.

```bash
python main.py -c paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="pedestrian_attribute_006.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_pedestrian_attribute.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
. Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.<b>Pipeline Integration</b>

The pedestrian attribute recognition module can be integrated into the [Pedestrian Attribute Recognition Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute_recognition.en.md) of PaddleX. Simply replace the model path to update the pedestrian attribute recognition module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the pedestrian attribute recognition module. Refer to the Python example code in <a href="#lable">Quick Integration</a>  and simply replace the model with the path to your trained model.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).
