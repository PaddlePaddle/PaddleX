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
<th>precision</th>
<th>recall</th>
<th>f1_score</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>DLinear_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_ad_pretrained.pdparams">Training Model</a></td>
<td>0.9898</td>
<td>0.9396</td>
<td>0.9640</td>
<td>72.8K</td>
<td>DLinear_ad is a simple, efficient, and easy-to-use model for time-series anomaly detection.</td>
</tr>
<tr>
<td>Nonstationary_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_ad_pretrained.pdparams">Training Model</a></td>
<td>0.9855</td>
<td>0.8895</td>
<td>0.9257</td>
<td>1.5MB</td>
<td>Based on the transformer structure, this model is optimized for anomaly detection in non-stationary time series.</td>
</tr>
<tr>
<td>AutoEncoder_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/AutoEncoder_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/AutoEncoder_ad_pretrained.pdparams">Training Model</a></td>
<td>0.9936</td>
<td>0.8436</td>
<td>0.9127</td>
<td>32K</td>
<td>AutoEncoder_ad is a classic autoencoder-based model for efficient and easy-to-use time-series anomaly detection.</td>
</tr>
<tr>
<td>PatchTST_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_ad_pretrained.pdparams">Training Model</a></td>
<td>0.9878</td>
<td>0.9070</td>
<td>0.9459</td>
<td>164K</td>
<td>PatchTST is a high-precision time-series anomaly detection model that balances local patterns and global dependencies.</td>
</tr>
</tbody>
</table>

<b>Note: The above precision metrics are measured on the</b>PSM<b>dataset with a time-series input length of 100.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the time series anomaly detection module. You can switch models under this module freely, and you can also integrate the model inference of the time series anomaly detection module into your project. Before running the following code, please download the [demo csv](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv) to your local machine.

```bash
from paddlex import create_model
model = create_model(model_name="AutoEncoder_ad")
output = model.predict("ts_ad.csv", batch_size=1)
for res in output:
    res.print()
    res.save_to_csv(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': 'ts_ad.csv', 'anomaly':            label
timestamp
220226         1
220227         1
220228         0
220229         1
220230         1
...          ...
220317         1
220318         1
220319         1
220320         1
220321         1

[96 rows x 1 columns]}}
```

The meanings of the parameters in the running result are as follows:
- `input_path`: Indicates the path to the input time-series file for anomaly prediction.
- `anomaly`: Indicates the result of time-series anomaly detection. A value of 1 indicates a predicted anomaly, while 0 indicates a normal prediction. The prediction results can be saved as a CSV file using `res.save_to_csv()` and as a JSON file using `res.save_to_json()`.

Relevant methods, parameters, and explanations are as follows:

* `create_model` instantiates a time-series anomaly detection model (here, `TimesNet_cls` is used as an example). The detailed explanation is as follows:
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

* Note that `model_name` must be specified. After specifying `model_name`, the default PaddleX built-in model parameters will be used. If `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the time-series anomaly detection model is called for inference prediction. The parameters of the `predict()` method are `input` and `batch_size`, which are explained as follows:

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
<td>Data for prediction, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python Variable</b>, such as time-series data represented by <code>pandas.DataFrame</code></li>
  <li><b>File Path</b>, such as the local path of a time-series file: <code>/root/data/ts.csv</code></li>
  <li><b>URL Link</b>, such as the network URL of a time-series file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv">Example</a></li>
  <li><b>Local Directory</b>, the directory should contain data files for prediction, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, elements of the list must be of the above types of data, such as <code>[pandas.DataFrame, pandas.DataFrame]</code>, <code>["/root/data/ts1.csv", "/root/data/ts2.csv"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"ts": "/root/data1"}, {"ts": "/root/data2/ts.csv"}]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer greater than 0</td>
<td>1</td>
</tr>
</table>

* Process the prediction results. The prediction result for each sample is of type `dict`, and supports operations such as printing, saving as a `csv` file, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
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
<td>The file path for saving. When a directory is specified, the saved file name will match the input file name</td>
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
<td><code>save_to_csv()</code></td>
<td>Save the result as a CSV file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is specified, the saved file name will match the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining the visualization of the time-series results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>csv</code></td>
<td rowspan="1">Get the time-series anomaly detection prediction results in <code>csv</code> format</td>
</tr>
</table>

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

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).