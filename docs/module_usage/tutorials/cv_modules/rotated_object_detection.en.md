---
comments: true
---

# Rotated Object Detection Module Usage Tutorial

## I. Overview
Rotated object detection is a derivative of the object detection module, specifically designed for detecting rotated objects. Rotated bounding boxes (Rotated Bounding Boxes) are commonly used for detecting rectangles with angle information, where the width and height of the rectangle are no longer parallel to the image coordinate axes. Compared to horizontal bounding boxes, rotated bounding boxes generally include less background information. Rotated box detection is often used in scenarios such as remote sensing.

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-YOLOE-R-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b1_v2/PP-YOLOE-R-L_infer.tar">Inference Model</a>/<a href="https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams">Training Model</a></td>
<td>78.14</td>
<td>20.7039</td>
<td>157.942</td>
<td>211.0 M</td>
<td rowspan="1">PP-YOLOE-R is an efficient single-stage Anchor-free rotated box detection model. Based on PP-YOLOE, PP-YOLOE-R introduces a series of useful designs to improve detection accuracy with minimal parameters and computational cost.</td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are on the <a href="https://captain-whu.github.io/DOTA/">DOTA</a> validation set mAP(0.5:0.95)。All model GPU inference times are based on an NVIDIA TRX2080 Ti machine, with precision type F16, and CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.</b></p>
> ❗ The above listed are the rotated object detection models currently supported by paddleX，actually PaddleDetection supports<b>10</b>rotated object detection models, For a detailed model list, please refer to <a href="https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/rotate">PaddleDetection</a>


## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, please refer to [PaddleX Local Installation Tutorial](../../../installation/installation.en.md)

After completing the installation of the wheel package, a few lines of code can complete the inference of the rotated object detection module. You can switch models under this module at will, and you can also integrate the model inference of the rotated object detection module into your project. Before running the following code, please download the [sample image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png) to your local machine.

```python
from paddlex import create_model
model = create_model("PP-YOLOE-R-L")
output = model.predict("rotated_object_detection_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```bash
{'res': "{'input_path': 'rotated_object_detection_001.png', 'boxes': [{'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7513620853424072, 'coordinate': [92.72234, 763.36676, 84.7699, 749.9725, 116.207375, 731.8547, 124.15982, 745.2489]}, {'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7284387350082397, 'coordinate': [348.60703, 177.85127, 332.80432, 149.83975, 345.37347, 142.95677, 361.17618, 170.96828]}, {'cls_id': 11, 'label': 'roundabout', 'score': 0.7909174561500549, 'coordinate': [535.02216, 697.095, 201.49803, 608.4738, 292.2446, 276.9634, 625.76874, 365.5845]}]}"}
```

The meanings of the parameters in the running results are as follows:
- `input_path`: The path of the input image to be predicted.
- `boxes`: Information about each predicted object.
  - `cls_id`: Class ID.
  - `label`: Class name.
  - `score`: Prediction score.
  - `coordinate`: Coordinates of the predicted bounding box, in the format <code>[x1, y1, x2, y2, x3, y3, x4, y4]</code>.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/BluebirdStory/PaddleX_doc_images/main/images/modules/robj_det/rotated_object_detection_001_res.png" />

Note: Due to network issues, the parsing of the above URL may not have been successful. If you need the content of this webpage, please check the validity of the URL and try again.

Related methods and parameter explanations are as follows:

* `create_model` instantiates a rotated object detection model (using `PP-YOLOE-R_L` as an example). The specific explanations are as follows:
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
<td><code>None</code></td>
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
<td>The threshold for filtering low-score objects</td>
<td><code>float/None/dict</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>The resolution used by the model for prediction</td>
<td><code>int/tuple/None</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the model parameters built into PaddleX will be used by default. If `model_dir` is specified, the user-defined model will be used.

* `threshold` is the threshold for filtering low-score objects. The default is `None`, which means using the settings from the previous layer. The priority of parameter settings from high to low is: `predict parameter input > create_model initialization > yaml configuration file setting`. Currently, two threshold setting methods are supported:
  * `float`: Use the same threshold for all classes.
  * `dict`: The key is the class ID, and the value is the threshold, allowing different thresholds for different classes.

* `img_size` is the resolution used by the model for prediction. The default is `None`, which means using the settings from the previous layer. The priority of parameter settings from high to low is: `create_model initialization > yaml configuration file setting`.

* The `predict()` method of the rotated object detection model is called for inference prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `threshold`, with specific explanations as follows:

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
  <li><b>List</b>, the elements of the list must be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<td>The threshold for filtering low-score objects</td>
<td><code>float</code>/<code>dict</code>/<code>None</code></td>
<td>
<ul>
  <li><b>None</b>, indicating the use of settings from the previous layer. The priority of parameter settings from high to low is: <code>predict parameter input > create_model initialization > yaml configuration file setting</code></li>
  <li><b>float</b>, such as 0.5, indicating the use of <code>0.5</code> as the threshold for all classes during inference</li>
  <li><b>dict</b>, such as <code>{0: 0.5, 1: 0.35}</code>, indicating the use of 0.5 as the threshold for class 0 and 0.35 as the threshold for class 1 during inference</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* The prediction results are processed, and the prediction result of each sample is of type `dict`, supporting operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan = "3">Print the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the results as a file in JSON format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as a file in image format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more usage methods of the single model inference API in PaddleX, please refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Secondary Development
If you are pursuing higher accuracy with existing models, you can use PaddleX's secondary development capabilities to develop better rotated object detection models. Before using PaddleX to develop rotated object detection models, please ensure that you have installed the model training plugins related to rotated object detection in PaddleX. The installation process can be referred to [PaddleX Local Installation Tutorial](../../../installation/installation.en.md)

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data verification functionality for each module, only data that passes the verification can be used for model training. Additionally, PaddleX provides a Demo dataset for each module, which you can use to complete subsequent development. If you wish to use a private dataset for subsequent model training, you can refer to [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can refer to the following command to download the Demo dataset to the specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/rdet_dota_examples.tar -P ./dataset
tar -xf ./dataset/rdet_dota_examples.tar -C ./dataset/
```
After decompression, the dataset directory structure is as follows:：
```bash
- dataset/DOTA-sampled200_crop1024_data
  - annotations
    - instance_train.json
    - instance_val.json
  - images
    - img1.png
    - img2.png
    - img3.png
    ...
```
#### 4.1.2 Data Verification
A single command can complete data verification:

```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
```
After executing the above command, PaddleX will verify the dataset and count the basic information of the dataset. After the command runs successfully, the log will print `Check dataset passed !`. The verification result file is saved in `./output/check_dataset_result.json`, and the related outputs are saved in the`./output/check_dataset`directory under the current directory, including visualized sample images and sample distribution histograms.

<details><summary>👉 <b> Verification Result Details (Click to Expand)</b></summary>

<p>The specific content of the verification result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;num_classes&quot;: 15,
    &quot;train_samples&quot;: 1892,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/P2610__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1137__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1122__1.0__5888___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P0543__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P0518__1.0__0___91.png&quot;,
      &quot;check_dataset\/demo_img\/P0961__1.0__1648___87.png&quot;,
      &quot;check_dataset\/demo_img\/P1732__1.0__0___824.png&quot;,
      &quot;check_dataset\/demo_img\/P2766__1.0__4421___0.png&quot;,
      &quot;check_dataset\/demo_img\/P2582__1.0__674___725.png&quot;,
      &quot;check_dataset\/demo_img\/P1529__1.0__2976___1648.png&quot;
    ],
    &quot;val_samples&quot;: 473,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/P2342__1.0__890___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1386__1.0__2472___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P0961__1.0__824___87.png&quot;,
      &quot;check_dataset\/demo_img\/P1651__1.0__824___824.png&quot;,
      &quot;check_dataset\/demo_img\/P1529__1.0__824___2976.png&quot;,
      &quot;check_dataset\/demo_img\/P0961__1.0__4944___87.png&quot;,
      &quot;check_dataset\/demo_img\/P0725__1.0__634___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1679__1.0__1648___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P2726__1.0__824___1578.png&quot;,
      &quot;check_dataset\/demo_img\/P0457__1.0__379___0.png&quot;,
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;rdet_dota_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;COCODetDataset&quot;
}
</code></pre>
<p>In the above verification result, check_pass is true, indicating that the dataset format meets the requirements. The explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>：The number of categories in this dataset is 15;</li>
<li><code>attributes.train_samples</code>：The number of training set samples in this dataset is 1892;</li>
<li><code>attributes.val_samples</code>：The number of validation set samples in this dataset is 473;</li>
<li><code>attributes.train_sample_paths</code>：The relative path list of visualized training set sample images in this dataset;</li>
<li><code>attributes.val_sample_paths</code>：The relative path list of visualized validation set sample images in this dataset;</li>
</ul>
<p>Additionally, the dataset verification also analyzes the distribution of sample quantities for all categories in the dataset and draws a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/rotated_object_detection/01.png"></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing the data verification, you can convert the dataset format or re-split the training/validation ratio of the dataset by modifying the configuration file or adding hyperparameters.

<details><summary>👉 <b>Format Conversion/Dataset Splitting Details (Click to Expand)）</b></summary>

<p><b>（1）Dataset Format Conversion</b></p>

Rotated object detection does not support dataset format conversion, only standard <b>DOTA COCO data format</b>
<p><b>（2）Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example explanations for the parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset, set to <code>True</code> to convert the dataset format, default is <code>False</code>；</li>
<li><code>train_percent</code>: If re-splitting the dataset, you need to set the percentage of the training set, which is any integer between 0-100, and needs to ensure that the sum with <code>val_percent</code> is 100；</li>
<li><code>val_percent</code>: If re-splitting the dataset, you need to set the percentage of the validation set, which is any integer between 0-100, and needs to ensure that the sum with <code>train_percent</code> is 100;
For example, if you want to re-split the dataset with 90% for the training set and 10% for the validation set, you need to modify the configuration file as follows:
</li>
</ul>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
</code></pre>
<p>After the dataset splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code>.</p>
<p>The above parameters also support setting through adding command line parameters:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
A single command can complete model training, taking the training of the rotated object detection model `PP-YOLOE-R-L` as an example:

```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
```
The following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-YOLOE-R-L.yaml`. When training other models, you need to specify the corresponding configuration file. The correspondence between models and configuration files can be found in [PaddleX Model List (CPU/GPU))](../../../support_list/models_list.en.md)）
* Specify the mode as model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under Global and Train in the `.yaml` configuration file, or by adding parameters in the command line. For example, specify the first 2 GPU cards for training: `-o Global.device=gpu:0,1`; set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and detailed explanations, please refer to the configuration file instructions for the corresponding task module [PaddleX Common Model Configuration File Parameter Instructions.](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Explanations (Click to Expand)</b></summary>

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
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-YOLOE-R-L.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference and Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference

* To perform inference predictions through the command line, use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png) to your local machine.
```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="rotated_object_detection_001.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-YOLOE-R-L.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own project.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the object detection module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.
