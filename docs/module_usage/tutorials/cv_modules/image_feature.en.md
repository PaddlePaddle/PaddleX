---
comments: true
---

# Image Feature Module Development Tutorial

## I. Overview
The image feature module is one of the important tasks in computer vision, primarily referring to the automatic extraction of useful features from image data using deep learning methods, to facilitate subsequent image retrieval tasks. The performance of this module directly affects the accuracy and efficiency of the subsequent tasks. In practical applications, image features typically output a set of feature vectors, which can effectively represent the content, structure, texture, and other information of the image, and will be passed as input to the subsequent retrieval module for processing.

## II. Supported Model List


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recall@1 (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-ShiTuV2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_pretrained.pdparams">Trained Model</a></td>
<td>84.2</td>
<td>3.48 / 0.55</td>
<td>8.04 / 4.04</td>
<td>16.3 M</td>
<td rowspan="3">PP-ShiTuV2 is a general image feature system consisting of three modules: object detection, feature extraction, and vector retrieval. These models are part of the feature extraction module and can be selected based on system requirements.</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">Trained Model</a></td>
<td>88.69</td>
<td>12.94 / 2.88</td>
<td>58.36 / 58.36</td>
<td>306.6 M</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams">Trained Model</a></td>
<td>91.03</td>
<td>51.65 / 11.18</td>
<td>255.78 / 255.78</td>
<td>1.05 G</td>
</tr>
</table>
<b>Note: The above accuracy metrics are Recall@1 from AliProducts. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the image feature module. You can switch between models under this module freely, and you can also integrate the model inference of the image feature module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "PP-ShiTuV2_rec"
model = create_model(model_name)
output = model.predict("general_image_recognition_001.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_json("./output/res.json")
```

After running, the result is:

```bash
{'res': {'input_path': 'general_image_recognition_001.jpg', 'page_index': None, 'feature': array([ 0.04910231, ..., -0.07126085], dtype=float32)}}
```

The meanings of the parameters are as follows:
- `input_path`: The path to the input image to be predicted.
- `feature`: The extracted image feature vector, with a dimensionality equal to the model's output feature dimension, which is 512 in this case.

Descriptions of related methods, parameters, etc., are as follows:

* The `create_model` method instantiates an image feature model (using `PP-ShiTuV2_rec` as an example). Specific descriptions are as follows:
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
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>The storage path of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. When `model_name` is specified, PaddleX's built-in model parameters are used by default. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the image feature model is called for inference. The parameters of the `predict()` method are `input` and `batch_size`, with specific descriptions as follows:

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
<td>Any integer</td>
<td>1</td>
</tr>
</table>

* Process the prediction results. Each sample's prediction result is of type `dict`, and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Explanation</th>
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
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a <code>json</code> file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is provided, the saved file name matches the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

* Additionally, the prediction results can be accessed via properties, as follows:

<table>
<thead>
<tr>
<th>Property</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td><code>json</code></td>
<td>Get the prediction result in <code>json</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference APIs, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better image feature models. Before developing image feature models with PaddleX, ensure you have installed the classification-related model training plugins for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>.  Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Image Feature Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/image_feature.en.md).


#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:
```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Inshop_examples.tar -P ./dataset
tar -xf ./dataset/Inshop_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
A single command can complete data validation:
```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/05_1_front.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/04_3_back.jpg",
      "check_dataset/demo_img/04_2_side.jpg",
      "check_dataset/demo_img/12_1_front.jpg",
      "check_dataset/demo_img/07_2_side.jpg",
      "check_dataset/demo_img/04_7_additional.jpg",
      "check_dataset/demo_img/04_4_full.jpg",
      "check_dataset/demo_img/01_1_front.jpg"
    ],
    "gallery_samples": 110,
    "gallery_sample_paths": [
      "check_dataset/demo_img/06_2_side.jpg",
      "check_dataset/demo_img/01_4_full.jpg",
      "check_dataset/demo_img/04_7_additional.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/02_4_full.jpg",
      "check_dataset/demo_img/03_4_full.jpg",
      "check_dataset/demo_img/02_2_side.jpg",
      "check_dataset/demo_img/03_2_side.jpg"
    ],
    "query_samples": 125,
    "query_sample_paths": [
      "check_dataset/demo_img/08_7_additional.jpg",
      "check_dataset/demo_img/01_7_additional.jpg",
      "check_dataset/demo_img/02_4_full.jpg",
      "check_dataset/demo_img/04_4_full.jpg",
      "check_dataset/demo_img/09_7_additional.jpg",
      "check_dataset/demo_img/04_3_back.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/06_2_side.jpg",
      "check_dataset/demo_img/02_7_additional.jpg",
      "check_dataset/demo_img/02_2_side.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/Inshop_examples",
  "show_type": "image",
  "dataset_type": "ShiTuRecDataset"
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:
* <code>attributes.train_samples</code>: The number of training samples in this dataset is 1000;
* <code>attributes.gallery_samples</code>: The number of gallery (or reference) samples in this dataset is 110;
* <code>attributes.query_samples</code>: The number of query samples in this dataset is 125;
* <code>attributes.train_sample_paths</code>: A list of relative paths to the visual images of training samples in this dataset;
* <code>attributes.gallery_sample_paths</code>: A list of relative paths to the visual images of gallery (or reference) samples in this dataset;
* <code>attributes.query_sample_paths</code>: A list of relative paths to the visual images of query samples in this dataset;</p>
<p>Additionally, the dataset verification also analyzes the number of images and image categories within the dataset, and generates a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/img_recognition/01.png"/></p></details>

### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data verification, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>The image feature task supports converting <code>LabelMe</code> format datasets to <code>ShiTuRecDataset</code> format. The parameters for dataset format conversion can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example parameter descriptions in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion. The image feature task supports converting <code>LabelMe</code> format datasets to <code>ShiTuRecDataset</code> format, default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, the source dataset format needs to be set, default is <code>null</code>, optional value is <code>LabelMe</code>;</li>
</ul>
<p>For example, if you want to convert a <code>LabelMe</code> format dataset to <code>ShiTuRecDataset</code> format, you need to modify the configuration file as follows:</p>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/image_classification_labelme_examples.tar -P ./dataset
tar -xf ./dataset/image_classification_labelme_examples.tar -C ./dataset/
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples
</code></pre>
<p>After the data conversion is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example parameter descriptions in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When <code>True</code>, the dataset will be re-split, default is <code>False</code>;</li>
<li><code>train_percent</code>: If the dataset is re-split, the percentage of the training set needs to be set, the type is any integer between 0-100, and it needs to ensure that the sum of <code>gallery_percent</code> and <code>query_percent</code> values is 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with 70% training set, 20% gallery set, and 10% query set, you need to modify the configuration file as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 70
    gallery_percent: 20
    query_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=70 \
    -o CheckDataset.split.gallery_percent=20 \
    -o CheckDataset.split.query_percent=10
</code></pre>
<blockquote>
<p>❗Note: Due to the specificity of image feature model evaluation, data partitioning is meaningful only when the train, query, and gallery sets belong to the same category system. During the evaluation of recognition models, it is imperative that the gallery and query sets belong to the same category system, which may or may not be the same as the train set. If the gallery and query sets do not belong to the same category system as the train set, the evaluation after data partitioning becomes meaningless. It is recommended to proceed with caution.</p>
</blockquote></details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of the image feature model PP-ShiTuV2_rec as an example:

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`，When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module of the model [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).

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

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file, detailed instructions can be found in [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including recall1、recall5、mAP；</p></details>

### <b>4.4 Model Inference and Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.


#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg) to your local machine.

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_image_recognition_001.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`.
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

> ❗ Note: The inference result of the recognition model is a set of vectors, which requires a retrieval module to complete image feature.

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.<b>Pipeline Integration</b>

The image feature module can be integrated into the <b>General Image Recognition Pipeline</b> (comming soon) of PaddleX. Simply replace the model path to update the image feature module of the relevant pipeline. In pipeline integration, you can use service-oriented deployment to deploy your trained model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the image feature module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).