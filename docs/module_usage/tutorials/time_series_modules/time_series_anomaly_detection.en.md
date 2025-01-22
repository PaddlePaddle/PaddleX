---
comments: true
---

# Time Series Anomaly Detection Module Development Tutorial

## I. Overview
Time series anomaly detection focuses on identifying abnormal points or periods in time series data that do not conform to expected patterns, trends, or periodic regularities. These anomalies can be caused by system failures, external shocks, data entry errors, or rare events, and are of great significance for timely response, risk assessment, and business decision-making.

## II. Supported Model List

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-Score</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>AutoEncoder_ad_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/AutoEncoder_ad_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/AutoEncoder_ad_ad_pretrained.pdparams">Trained Model</a></td>
<td>0.9898</td>
<td>0.9396</td>
<td>0.9641</td>
<td>72.8K</td>
<td>AutoEncoder_ad_ad is a simple, efficient, and easy-to-use time series anomaly detection model</td>
</tr>
<tr>
<td>Nonstationary_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/Nonstationary_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_ad_pretrained.pdparams">Trained Model</a></td>
<td>0.9855</td>
<td>0.8895</td>
<td>0.9351</td>
<td>1.5MB</td>
<td>Based on the transformer structure, optimized for anomaly detection in non-stationary time series</td>
</tr>
<tr>
<td>AutoEncoder_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/AutoEncoder_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/AutoEncoder_ad_pretrained.pdparams">Trained Model</a></td>
<td>0.9936</td>
<td>0.8436</td>
<td>0.9125</td>
<td>32K</td>
<td>AutoEncoder_ad is a classic autoencoder-based, efficient, and easy-to-use time series anomaly detection model</td>
</tr>
<tr>
<td>PatchTST_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PatchTST_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_ad_pretrained.pdparams">Trained Model</a></td>
<td>0.9878</td>
<td>0.9070</td>
<td>0.9457</td>
<td>164K</td>
<td>PatchTST is a high-precision time series anomaly detection model that balances local patterns and global dependencies</td>
</tr>
<tr>
<td>TimesNet_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/TimesNet_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_ad_pretrained.pdparams">Trained Model</a></td>
<td>0.9837</td>
<td>0.9480</td>
<td>0.9656</td>
<td>732K</td>
<td>Through multi-period analysis, TimesNet is an adaptive and high-precision time series anomaly detection model</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the PSM dataset with a time series length of 100.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the time series anomaly detection module. You can switch models under this module freely, and you can also integrate the model inference of the time series anomaly detection module into your project. Before running the following code, please download the [demo csv](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv) to your local machine.

```bash
from paddlex import create_model
model = create_model("AutoEncoder_ad")
output = model.predict("ts_ad.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
For more information on using PaddleX's single model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better time series anomaly detection models. Before developing time series anomaly models with PaddleX, please ensure that the PaddleTS plugin is installed. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and only data that passes validation can be used for model training. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Time Series Anomaly Detection Task Module Data Annotation Tutorial](../../../data_annotations/time_series_modules/time_series_anomaly_detection.en.md).

#### 4.1.1 Demo Data Download
You can use the following command to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar -P ./dataset
tar -xf ./dataset/ts_anomaly_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:
```bash
python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```
After executing the above command, PaddleX will validate the dataset, summarize its basic information, and print `Check dataset passed !` in the log if the command runs successfully. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the current directory's `./output/check_dataset` directory, including example time series data.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 22032,
    &quot;train_table&quot;: [
      [
        &quot;timestamp&quot;,
        &quot;feature_0&quot;,
        &quot;...&quot;,
        &quot;feature_24&quot;,
        &quot;label&quot;
      ],
      [
        0.0,
        0.7326893750079723,
        &quot;...&quot;,
        0.1382488479262673,
        0.0
      ]
    ],
    &quot;val_samples&quot;: 198290,
    &quot;val_table&quot;: [
      [
        &quot;timestamp&quot;,
        &quot;feature_0&quot;,
        &quot;...&quot;,
        &quot;feature_24&quot;,
        &quot;label&quot;
      ],
      [
        22032.0,
        0.8604795809835284,
        &quot;...&quot;,
        0.1428571428571428,
        0.0
      ]
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/ts_anomaly_examples&quot;,
  &quot;show_type&quot;: &quot;csv&quot;,
  &quot;dataset_type&quot;: &quot;TSADDataset&quot;
}
</code></pre>
<p>The verification results above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 22032;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 198290;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the top 10 rows of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the top 10 rows of validation samples in this dataset.
<b>Note</b>: Only data that has passed validation can be used for training and evaluation.</li>
</ul></details>


### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data validation, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.


<details><summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Time series anomaly detection supports converting <code>xlsx</code> and <code>xls</code> format datasets to <code>csv</code> format.</p>
<p>Parameters related to dataset validation can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example parameter descriptions in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to convert the dataset format, supporting <code>xlsx</code> and <code>xls</code> formats to <code>CSV</code> format, default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, the source dataset format does not need to be set, default is <code>null</code>;</li>
</ul>
<p>To enable format conversion, modify the configuration as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: null
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
</code></pre>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples \
    -o CheckDataset.convert.enable=True
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters related to dataset validation can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example parameter descriptions in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to convert the dataset format, <code>True</code> to enable dataset format conversion, default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, time series anomaly detection only supports converting xlsx annotation files to csv, the source dataset format does not need to be set, default is <code>null</code>;</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset, <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set, an integer between 0-100, ensuring the sum with <code>val_percent</code> is 100;</li>
<li><code>val_percent</code>: If re-splitting the dataset, set the percentage of the validation set, an integer between 0-100, ensuring the sum with <code>train_percent</code> is 100;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with just one command. Here, we use the Time Series Forecasting model (AutoEncoder_ad) as an example:

```bash
python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `AutoEncoder_ad.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to train using the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX TS Configuration Parameters Documentation](../../instructions/config_parameters_time_series.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves model weight files, with the default path being <code>output</code>. To specify a different save path, use the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights are produced, and static graph weights are used by default for model inference.</li>
<li>
<p>After model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, including whether the training task completed successfully, produced weight metrics, and related file paths.</p>
</li>
<li><code>train.log</code>: Training log file, recording model metric changes, loss changes, etc.</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameters used for this training session.</li>
<li><code>best_accuracy.pdparams.tar</code>, <code>scaler.pkl</code>, <code>.checkpoints</code>, <code>.inference</code>: Model weight-related files, including Model weight-related files, including network parameters, optimizers, and network architecture.</li>
</ul></details>


### 4.3 Model Evaluation
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `AutoEncoder_ad.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../instructions/config_parameters_time_series.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, the following outputs are typically generated:</p>
<p>Upon completion of model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically indicating whether the evaluation task was completed successfully and the model's evaluation metrics, including <code>f1</code>, <code>recall</code>, and <code>precision</code>.</p></details>

### 4.4 Model Inference and Integration
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions through the command line, simply use the following command:

Before running the following code, please download the [demo csv](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv) to your local machine.

```bash
python main.py -c paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="ts_ad.csv"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `AutoEncoder_ad.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../instructions/config_parameters_time_series.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1. <b>Pipeline Integration</b>

The time series prediction module can be integrated into PaddleX pipelines such as [Time Series Anomaly Detection](../../../pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.en.md). Simply replace the model path to update the time series prediction model. In pipeline integration, you can use service deployment to deploy your obtained model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the time series anomaly detection module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), simply replace the model with the path to your trained model.
