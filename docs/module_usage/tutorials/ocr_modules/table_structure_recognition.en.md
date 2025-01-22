---
comments: true
---

# Table Structure Recognition Module Development Tutorial

## I. Overview
Table structure recognition is a crucial component in table recognition systems, converting non-editable table images into editable table formats (e.g., HTML). The goal of table structure recognition is to identify the rows, columns, and cell positions of tables. The performance of this module directly impacts the accuracy and efficiency of the entire table recognition system. The module typically outputs HTML or LaTeX code for the table area, which is then passed to the table content recognition module for further processing.

## II. Supported Model List


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Trained Model</a></td>
<td>59.52</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td rowspan="1">SLANet is a table structure recognition model developed by Baidu PaddlePaddle Vision Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>

<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Trained Model</a></td>
<td>63.69</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td rowspan="1">
SLANet_plus is an enhanced version of SLANet, a table structure recognition model developed by Baidu PaddlePaddle's Vision Team. Compared to SLANet, SLANet_plus significantly improves its recognition capabilities for wireless and complex tables, while reducing the model's sensitivity to the accuracy of table localization. Even when there are offsets in table localization, it can still perform relatively accurate recognition.
</td>
</tr>
</table>

<b>Note: The above accuracy metrics are evaluated on a self-built English table recognition dataset by PaddleX. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>


## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the table structure recognition module. You can easily switch models within this module and integrate the model inference into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your local machine.

```bash
from paddlex import create_model
model = create_model("SLANet")
output = model.predict("table_recognition.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better table structure recognition models. Before developing table structure recognition models with PaddleX, ensure you have installed the PaddleOCR plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to [PaddleX Table Structure Recognition Task Module Data Annotation Tutorial](../../../data_annotations/ocr_modules/table_recognition.en.md)

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following command:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar -P ./dataset
tar -xf ./dataset/table_rec_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
Run a single command to complete data validation:

```bash
python main.py -c paddlex/configs/modules/table_recognition/SLANet.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/table_rec_dataset_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 2000,
    &quot;train_sample_paths&quot;: [
      &quot;../dataset/table_rec_dataset_examples/images/border_right_7384_X9UFEPKVMLALY7DDB11A.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_top_13708_VE2DGBD4DCQU2ITLBTEA.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_top_6490_14Z6ZN6G52GG4XA0K4XU.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_top_14236_DG96EX0EDKIIDK8P6ENG.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_19648_SV8B7X34RTYRAT2T5CPI.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_bottom_7186_HODBC25HISMCSVKY0HJ9.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/head_border_bottom_5773_4K4H9OVK9X9YVHE4Y1BQ.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_7760_8C62CCH5T57QUGE0NTHZ.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_bottom_15707_B1YVOU3X4NHHB6TL269O.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/no_border_5223_HLG406UK35UD5EUYC2AV.jpg&quot;
    ],
    &quot;val_samples&quot;: 100,
    &quot;val_sample_paths&quot;: [
      &quot;../dataset/table_rec_dataset_examples/images/border_2945_L7MSRHBZRW6Y347G39O6.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/head_border_bottom_4825_LH9WI6X104CP3VFXPSON.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/head_border_bottom_16837_79KHWU9WDM9ZQHNBGQAL.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_bottom_10107_9ENLLC29SQ6XI8WZY53E.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_top_16668_JIS0YFDZKTKETZIEKCKX.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_18653_J9SSKHLFTRJD4J8W17OW.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_bottom_8396_VJ3QJ3I0DP63P4JR77FE.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_9017_K2V7QBWSU2BA4R3AJSO7.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/border_top_19494_SDFMWP92NOB2OT7109FI.jpg&quot;,
      &quot;../dataset/table_rec_dataset_examples/images/no_border_288_6LK683JUCMOQ38V5BV29.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {},
  &quot;dataset_path&quot;: &quot;./dataset/table_rec_dataset_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;PubTabTableRecDataset&quot;
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of samples in the training set of this dataset is 2000;</li>
<li><code>attributes.val_samples</code>: The number of samples in the validation set of this dataset is 100;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualization images of samples in the training set of this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualization images of samples in the validation set of this dataset.</li>
</ul></details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Table structure recognition does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The dataset splitting parameters can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. An example of part of the configuration file is shown below:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, ensuring the sum with <code>val_percent</code> equals 100;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_recognition/SLANet.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/table_rec_dataset_examples
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in their original paths.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/table_recognition/SLANet.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/table_rec_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>


### 4.2 Model Training
A single command can complete the model training. Taking the training of the table structure recognition model SLANet as an example:

```bash
python main.py -c paddlex/configs/modules/table_recognition/SLANet.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/table_rec_dataset_examples
```
the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `SLANet.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
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

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:
```bash
python main.py -c paddlex/configs/modules/table_recognition/SLANet.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/table_rec_dataset_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it's `SLANet.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter to set it, such as <code>-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including acc ;</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
* Inference predictions can be performed through the command line with just one command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/modules/table_recognition/SLANet.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="table_recognition.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it's `SLANet.yaml `)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`
* Specify the input data path: `-o Predict.input="..."`. Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).
* Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.


#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.<b>Pipeline Integration</b>

The table structure recognition module can be integrated into PaddleX pipelines such as the [General Table Recognition Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition.en.md) and the [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3)](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.en.md). Simply replace the model path to update the table structure recognition module in the relevant pipelines. For pipeline integration, you can deploy your obtained model using high-performance inference and service-oriented deployment.

2.<b>Module Integration</b>

The model weights you produce can be directly integrated into the table structure recognition module. Refer to the Python example code in [Quick Integration](#iii-quick-integration) , and simply replace the model with the path to your trained model.
