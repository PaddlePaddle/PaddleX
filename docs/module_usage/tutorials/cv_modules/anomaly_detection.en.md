---
comments: true
---

# Unsupervised Anomaly Detection Module Development Tutorial

## I. Overview
Unsupervised anomaly detection is a technology that automatically identifies and detects anomalies or rare samples that are significantly different from the majority of data in a dataset, without labels or with a small amount of labeled data. This technology is widely used in many fields such as industrial manufacturing quality control and medical diagnosis.

## II. Supported Model List


<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>ROCAUC（Avg）</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>STFPM</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/STFPM_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/STFPM_pretrained.pdparams">Trained Model</a></td>
<td>0.962</td>
<td>22.5</td>
<td>An unsupervised anomaly detection algorithm based on representation consists of a pre-trained teacher network and a student network with the same structure. The student network detects anomalies by matching its own features with the corresponding features in the teacher network.</td>
</tr>
</tbody>
</table>
<b>The above model accuracy indicators are measured from the MVTec_AD dataset.</b>

## III. Quick Integration  <a id="quick"> </a>
Before quick integration, you need to install the PaddleX wheel package. For the installation method of the wheel package, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). After installing the wheel package, a few lines of code can complete the inference of the unsupervised anomaly detection module. You can switch models under this module freely, and you can also integrate the model inference of the unsupervised anomaly detection module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png) to your local machine.

```python
from paddlex import create_model

model_name = "STFPM"

model = create_model(model_name=model_name)
output = model.predict("uad_grid.png", batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```
After running, the result obtained is:

```bash
{'res': "{'input_path': 'uad_grid.png', 'pred': '...'}"}
```

The meanings of the running result parameters are as follows:
- `input_path`: Indicates the path of the input image to be detected for anomalies.
- `doctr_img`: Indicates the visualization result of the anomaly detection image. Since there is too much data to print directly, `...` is used here as a placeholder. The prediction result can be saved as an image through `res.save_to_img()` and as a JSON file through `res.save_to_json()`.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/image_ad/uad_grid_res.png">

Relevant methods, parameters, and explanations are as follows:

* `create_model` instantiates an image anomaly detection model (STFPM is used as an example here). The specific explanation is as follows:
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
<td>All model names supported by PaddleX</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX will be used. If `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the image anomaly detection model is called for inference prediction. The parameters of the `predict()` method are `input` and `batch_size`, with specific explanations as follows:

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
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
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

* The prediction results are processed, with each sample's prediction result being of type `dict`, and supporting operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
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
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
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
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining visualized images with results and prediction results through attributes, as follows:

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
<td rowspan="1">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

For more information on the usage of PaddleX's single-model inference API, please refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better unsupervised anomaly detection models. Before using PaddleX to develop unsupervised anomaly detection models, ensure you have installed the PaddleDetection plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development based on the official demos. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Semantic Segmentation Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/semantic_segmentation.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/mvtec_examples.tar -P ./dataset
tar -xf ./dataset/mvtec_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/image_anomaly_detection/STFPM.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mvtec_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/000.png&quot;,
      &quot;check_dataset/demo_img/001.png&quot;,
      &quot;check_dataset/demo_img/002.png&quot;
    ],
    &quot;train_samples&quot;: 264,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset/demo_img/000.png&quot;,
      &quot;check_dataset/demo_img/001.png&quot;,
      &quot;check_dataset/demo_img/002.png&quot;
    ],
    &quot;val_samples&quot;: 57,
    &quot;num_classes&quot;: 231
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;mvtec_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;SegDataset&quot;
}
</code></pre>
<p>The verification results mentioned above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Details of other indicators are as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 264;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 57;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of validation samples in this dataset;</li>
</ul></details>


### 4.2 Model Training

A single command is sufficient to complete model training, taking the training of STFPM as an example:

```bash
python main.py -c paddlex/configs/modules/image_anomaly_detection/STFPM.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/mvtec_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file of the model (here it is `STFPM.yaml`，When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`

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
python main.py -c paddlex/configs/modules/image_anomaly_detection/STFPM.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/mvtec_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `STFPM.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction. In PaddleX, model inference prediction can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png) to your local machine.
```bash
python main.py -c paddlex/configs/modules/image_anomaly_detection/STFPM.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="uad_grid.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `STFPM.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. <b>Pipeline Integration</b>

The unsupervised anomaly detection module can be integrated into PaddleX pipelines such as [Image_anomaly_detection](../../../pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.en.md). Simply replace the model path to update the unsupervised anomaly detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the unsupervised anomaly detection module. You can refer to the Python example code in [Quick Integration](#quick), simply replace the model with the path to your trained model.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).
