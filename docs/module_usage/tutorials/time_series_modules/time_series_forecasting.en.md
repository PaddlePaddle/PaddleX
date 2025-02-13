---
comments: true
---

# Time Series Forecasting Module Development Tutorial

## I. Overview
Time series forecasting aims to predict the possible values or states at a future point in time or within a future time period by analyzing patterns, trends, periodicity, and other characteristics in historical data. This helps enterprises and organizations make more accurate decisions, optimize resource allocation, reduce risks, and seize potential market opportunities. These time series data typically originate from various sensors, economic activities, social behaviors, and other real-world application scenarios. For example, stock prices, temperature changes, website traffic, sales data, and the like are all typical examples of time series data.

## II. Supported Model List

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mse</th>
<th>mae</th>
<th>Model Size (M)</th>
<th>Introduce</th>
</tr>
</thead>
<tbody>
<tr>
<td>DLinear</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_pretrained.pdparams">Trained Model</a></td>
<td>0.382</td>
<td>0.394</td>
<td>76k</td>
<td>Simple structure, high efficiency and easy-to-use time series prediction model</td>
</tr>
<tr>
<td>Nonstationary</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_pretrained.pdparams">Trained Model</a></td>
<td>0.600</td>
<td>0.515</td>
<td>60.3M</td>
<td>Based on the transformer structure, targeted optimization of long-term time series prediction models for non-stationary time series</td>
</tr>
<tr>
<td>PatchTST</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_pretrained.pdparams">Trained Model</a></td>
<td>0.385</td>
<td>0.397</td>
<td>2.2M</td>
<td>High-precision long-term time series prediction model that takes into account both local patterns and global dependencies</td>
</tr>
<tr>
<td>TiDE</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TiDE_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TiDE_pretrained.pdparams">Trained Model</a></td>
<td>0.405</td>
<td>0.412</td>
<td>34.9M</td>
<td>High-precision model suitable for handling multivariate, long-term time series prediction problems</td>
</tr>
<tr>
<td>TimesNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_pretrained.pdparams">Trained Model</a></td>
<td>0.417</td>
<td>0.431</td>
<td>5.2M</td>
<td>Through multi-period analysis, TimesNet is a highly adaptable high-precision time series analysis model</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the [ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar) test dataset, with an input sequence length of 96, and a prediction sequence length of 96 for all models except TiDE, which has a prediction sequence length of 720.</b>


## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)


Just a few lines of code can complete the inference of the Time Series Forecasting module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the Time Series Forecasting module into your project. Before running the following code, please download the [demo csv](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv) to your local machine.

```bash
from paddlex import create_model
model = create_model(model_name="DLinear")
output = model.predict("ts_fc.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': 'ts_fc.csv', 'forecast':                            OT
date
2018-06-26 20:00:00  9.586131
2018-06-26 21:00:00  9.379762
2018-06-26 22:00:00  9.252275
2018-06-26 23:00:00  9.249993
2018-06-27 00:00:00  9.164998
...                       ...
2018-06-30 15:00:00  8.830340
2018-06-30 16:00:00  9.291553
2018-06-30 17:00:00  9.097666
2018-06-30 18:00:00  8.905430
2018-06-30 19:00:00  8.993793

[96 rows x 1 columns]}}
```

The meanings of the parameters in the running results are as follows:
- `input_path`: Indicates the path to the input time-series file for prediction.
- `forecast`: Indicates the time-series forecast result. You can save the forecast results to a CSV file using `res.save_to_csv()` or to a JSON file using `res.save_to_json()`.

Descriptions of related methods, parameters, etc., are as follows:

* The `create_model` method instantiates a time-series forecasting model (using `DLinear` as an example). Specific descriptions are as follows:
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
<td>All model names supported by PaddleX</td>
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

* The `model_name` must be specified. After specifying `model_name`, PaddleX's built-in model parameters are used by default. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the time-series forecasting model is called for inference prediction. The parameters of the `predict()` method are `input` and `batch_size`, with specific descriptions as follows:

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
  <li><b>Python variable</b>, such as time-series data represented by <code>pandas.DataFrame</code></li>
  <li><b>File path</b>, such as the local path of a time-series file: <code>/root/data/ts.csv</code></li>
  <li><b>URL link</b>, such as the network URL of a time-series file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv">Example</a></li>
  <li><b>Local directory</b>, which must contain data files for prediction, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, elements of the list must be of the above types, such as <code>[pandas.DataFrame, pandas.DataFrame]</code>, <code>[\"/root/data/ts1.csv\", \"/root/data/ts2.csv\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code>, <code>[{\"ts\": \"/root/data1\"}, {\"ts\": \"/root/data2/ts.csv\"}]</code></li>
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

* The prediction results are processed for each sample, with the prediction result being of type `dict`, and support operations such as printing, saving as a `csv` file, and saving as a `json` file:

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
<tr>
<td><code>save_to_csv()</code></td>
<td>Save the result as a time-series <code>csv</code> file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is provided, the saved file name matches the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining the visualized time-series with results and the prediction results via attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>csv</code></td>
<td rowspan="1">Get the time-series prediction result in <code>csv</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development

If you seek higher accuracy, you can leverage PaddleX's custom development capabilities to develop better Time Series Forecasting models. Before developing a Time Series Forecasting model with PaddleX, ensure you have installed PaddleClas plugin for PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Dataset Preparation

Before model training, you need to prepare a dataset for the task. PaddleX provides data validation functionality for each module. <b>Only data that passes validation can be used for model training.</b> Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Time Series Forecasting Task Module Data Preparation Tutorial](../../../data_annotations/time_series_modules/time_series_forecasting.en.md).

#### 4.1.1 Demo Data Download

You can download the demo dataset to a specified folder using the following commands:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ts_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Once the command runs successfully, a message saying `Check dataset passed !` will be printed in the log. The verification results will be saved in `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory, including visual examples of sample images and a histogram of sample distribution.

<details><summary>👉 <b>Verification Result Details (click to expand)</b></summary>

<p>The specific content of the verification result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 12194,
    &quot;train_table&quot;: [
      [
        &quot;date&quot;,
        &quot;HUFL&quot;,
        &quot;HULL&quot;,
        &quot;MUFL&quot;,
        &quot;MULL&quot;,
        &quot;LUFL&quot;,
        &quot;LULL&quot;,
        &quot;OT&quot;
      ],
      [
        &quot;2016-07-01 00:00:00&quot;,
        5.827000141143799,
        2.009000062942505,
        1.5989999771118164,
        0.4620000123977661,
        4.203000068664552,
        1.3400000333786009,
        30.5310001373291
      ],
      [
        &quot;2016-07-01 01:00:00&quot;,
        5.692999839782715,
        2.075999975204468,
        1.4919999837875366,
        0.4259999990463257,
        4.142000198364259,
        1.371000051498413,
        27.78700065612793
      ]
    ],
    &quot;val_samples&quot;: 3484,
    &quot;val_table&quot;: [
      [
        &quot;date&quot;,
        &quot;HUFL&quot;,
        &quot;HULL&quot;,
        &quot;MUFL&quot;,
        &quot;MULL&quot;,
        &quot;LUFL&quot;,
        &quot;LULL&quot;,
        &quot;OT&quot;
      ],
      [
        &quot;2017-11-21 02:00:00&quot;,
        12.994000434875488,
        4.889999866485597,
        10.055999755859377,
        2.878000020980835,
        2.559000015258789,
        1.2489999532699585,
        4.7129998207092285
      ],
      [
        &quot;2017-11-21 03:00:00&quot;,
        11.92199993133545,
        4.554999828338623,
        9.097000122070312,
        3.0920000076293945,
        2.559000015258789,
        1.2790000438690186,
        4.8540000915527335
      ]
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/ts_dataset_examples&quot;,
  &quot;show_type&quot;: &quot;csv&quot;,
  &quot;dataset_type&quot;: &quot;TSDataset&quot;
}
</code></pre></details>

The verification results above indicate that `check_pass` being `True` means the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.train_samples`: The number of training samples in this dataset is 12194;
* `attributes.val_samples`: The number of validation samples in this dataset is 3484;
* `attributes.train_sample_paths`: A list of relative paths to the top 10 rows of training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the top 10 rows of validation samples in this dataset;


</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional) (Click to Expand)
<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

<p>After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by modifying the configuration file or appending hyperparameters.</p>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Time Series Forecasting supports converting <code>xlsx</code> and <code>xls</code> format datasets to the required format.</p>
<p>Parameters related to dataset verification can be set by modifying the <code>CheckDataset</code> fields in the configuration file. Example explanations for some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to enable dataset format conversion, supporting <code>xlsx</code> and <code>xls</code> format conversion, default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is enabled, the source dataset format needs to be set, default is <code>null</code>.</li>
</ul>
<p>Modify the <code>paddlex/configs/modules/ts_forecast/DLinear.yaml</code> configuration as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: null
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_forecast_to_convert
</code></pre>
<p>Of course, the above parameters also support being set by appending command-line arguments. For a <code>LabelMe</code> format dataset, the command is:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_forecast_to_convert \
    -o CheckDataset.convert.enable=True \
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> fields in the configuration file. Example explanations for some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to enable re-splitting the dataset, set to <code>True</code> to perform dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set, which should be an integer between 0 and 100, ensuring the sum with <code>val_percent</code> is 100;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training

Model training can be completed with just one command. Here, we use the Time Series Forecasting model (DLinear) as an example:

```bash
python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `DLinear.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
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
After model training, you can evaluate the specified model weights on the validation set to verify model accuracy. Using PaddleX for model evaluation requires just one command:

```bash
python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```

Similar to model training, follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `DLinear.yaml`).
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the validation dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For more details, refer to the [PaddleX TS Configuration Parameters Documentation](../../instructions/config_parameters_time_series.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter, e.g., <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After model evaluation, the following outputs are typically produced:</p>
<ul>
<li><code>evaluate_result.json</code>: Records the evaluation results, specifically whether the evaluation task completed successfully and the model's evaluation metrics, including <code>mse</code> and <code>mae</code>.</li>
</ul></details>

### 4.4 Model Inference and Integration
After model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions via the command line, use the following command:

Before running the following code, please download the [demo csv](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv) to your local machine.

```bash
python main.py -c paddlex/configs/modules/ts_forecast/DLinear.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="ts_fc.csv"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `DLinear.yaml`)

* Set the mode to model inference prediction: `-o Global.mode=predict`

* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`

Specify the input data path: `-o Predict.inputh="..."` Other related parameters can be set by modifying the fields under Global and Predict in the `.yaml` configuration file. For details, refer to PaddleX Common Model Configuration File Parameter Description.

Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration

The model can be directly integrated into the PaddleX pipeline or into your own projects.

1. <b>Pipeline Integration</b>

The Time Series Forecasting module can be integrated into PaddleX pipelines such as the [Time Series Forecasting Pipeline (ts_fc)](../../../pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.en.md). Simply replace the model path to update the Time Series Forecasting module's model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the Time Series Forecasting module. You can refer to the Python sample code in [Quick Integration](#iii-quick-integration) and just replace the model with the path to the model you trained.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).