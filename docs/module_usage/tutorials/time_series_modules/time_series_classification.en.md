---
comments: true
---

# Time Series Classification Module Development Tutorial

## I. Overview
Time series classification involves identifying and categorizing different patterns in time series data by analyzing trends, periodicity, seasonality, and other factors that vary over time. This technique is widely used in medical diagnosis and other fields, effectively classifying key information in time series data to provide robust support for decision-making.

## II. Supported Model List

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Acc(%)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>TimesNet_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/TimesNet_cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_cls_pretrained.pdparams">Trained Model</a></td>
<td>87.5</td>
<td>792K</td>
<td>TimesNet is an adaptive and high-accuracy time series classification model through multi-period analysis</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation set for the above accuracy metrics is UWaveGestureLibrary.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, you can perform inference for the time series classification module with just a few lines of code. You can switch models under this module freely, and you can also integrate the model inference of the time series classification module into your project. Before running the following code, please download the [demo csv](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv) to your local machine.

```bash
from paddlex import create_model
model = create_model("TimesNet_cls")
output = model.predict("ts_cls.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
For more information on using PaddleX's single-model inference APIs, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you aim for higher accuracy with existing models, you can leverage PaddleX's custom development capabilities to develop better time series classification models. Before using PaddleX to develop time series classification models, ensure you have installed the PaddleTS plugin. Refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md) for the installation process.

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for subsequent model training, refer to [PaddleX Time Series Classification Task Module Data Annotation Tutorial](../../../data_annotations/time_series_modules/time_series_classification.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_classify_examples.tar -P ./dataset
tar -xf ./dataset/ts_classify_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:

```bash
python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```
After executing the above command, PaddleX will validate the dataset, summarize its basic information, and print `Check dataset passed !` in the log if the command runs successfully. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the current directory's `./output/check_dataset` directory, including example time series data and class distribution histograms.



<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 82620,
    &quot;train_table&quot;: [
      [
        &quot;Unnamed: 0&quot;,
        &quot;group_id&quot;,
        &quot;dim_0&quot;,
        ...,
        &quot;dim_60&quot;,
        &quot;label&quot;,
        &quot;time&quot;
      ],
      [
        0.0,
        0.0,
        0.000949,
        ...,
        0.12107,
        1.0,
        0.0
      ]
    ],
    &quot;val_samples&quot;: 83025,
    &quot;val_table&quot;: [
      [
        &quot;Unnamed: 0&quot;,
        &quot;group_id&quot;,
        &quot;dim_0&quot;,
        ...,
        &quot;dim_60&quot;,
        &quot;label&quot;,
        &quot;time&quot;
      ],
      [
        0.0,
        0.0,
        0.004578,
        ...,
        0.15728,
        1.0,
        0.0
      ]
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/ts_classify_examples&quot;,
  &quot;show_type&quot;: &quot;csv&quot;,
  &quot;dataset_type&quot;: &quot;TSCLSDataset&quot;
}
</code></pre>
<p>The verification results above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 12194;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 3484;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the top 10 rows of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the top 10 rows of validation samples in this dataset;</li>
</ul>
<p>Furthermore, the dataset validation also involved an analysis of the distribution of sample numbers across all categories within the dataset, and a distribution histogram (histogram.png) was generated accordingly.</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/time_classification/01.png"></p>
<p><b>Note</b>: Only data that has passed validation can be used for training and evaluation.</p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Time-series classification supports converting <code>xlsx</code> and <code>xls</code> format datasets to <code>csv</code> format.</p>
<p>Parameters related to dataset validation can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion, supporting conversion from <code>xlsx</code> and <code>xls</code> formats to <code>CSV</code> format, default is <code>False</code>;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
</code></pre>
<p>The above parameters can also be set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples \
    -o CheckDataset.convert.enable=True
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters related to dataset validation can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion, <code>True</code> to enable, default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, time-series classification only supports converting xlsx annotation files to csv, the source dataset format does not need to be set, default is <code>null</code>;</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset, <code>True</code> to enable, default is <code>False</code>;</li>
<li><code>train_percent</code>: If the dataset is re-split, the percentage of the training set needs to be set, an integer between 0-100, ensuring the sum with <code>val_percent</code> is 100;</li>
<li><code>val_percent</code>: If the dataset is re-split, the percentage of the validation set needs to be set, an integer between 0-100, ensuring the sum with <code>train_percent</code> is 100;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>


### 4.2 Model Training

Model training can be completed with just one command. Here, we use the Time Series Forecasting model (TimesNet_cls) as an example:

```bash
python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `TimesNet_cls.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
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
python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `TimesNet_cls.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other relevant parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../instructions/config_parameters_time_series.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, typically, the following outputs are generated:</p>
<p>Upon completion of model evaluation, an <code>evaluate_result.json</code> file is produced, which records the evaluation results, specifically whether the evaluation task was completed successfully and the model's evaluation metrics, including Top-1 Accuracy and F1 score.</p></details>

### 4.4 Model Inference and Model Integration
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction via the command line, simply use the following command:

Before running the following code, please download the [demo csv](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv) to your local machine.

```bash
python main.py -c paddlex/configs/modules/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="ts_cls.csv"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `TimesNet_cls.yaml` - <b>Note</b>: This should likely be `TimesNet_cls.yaml` for consistency)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other relevant parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../../module_usage/instructions/config_parameters_time_series.en.md).

#### 4.4.2 Model Integration
Models can be directly integrated into the PaddleX pipeline or directly into your own projects.

1. <b>Pipeline Integration</b>

The time series prediction module can be integrated into PaddleX pipelines such as [Time Series Classification](../../../pipeline_usage/tutorials/time_series_pipelines/time_series_classification.en.md). Simply replace the model path to update the time series prediction model. In pipeline integration, you can use service deployment to deploy your trained model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the time series classification module. Refer to the Python example code in [Quick Integration](#iii-quick-integration) (Note: This section header is in Chinese and should be translated or removed for consistency), simply replace the model with the path to your trained model.
