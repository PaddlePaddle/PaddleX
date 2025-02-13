---
comments: true
---

# PaddleX 3.0 Time Series Anomaly Detection Pipeline — Equipment Anomaly Detection Application Tutorial

PaddleX offers a rich set of pipelines, each consisting of one or more models that can solve specific scenario-based tasks. All PaddleX pipelines support quick trials, and if the results do not meet expectations, fine-tuning the models with private data is also supported. PaddleX provides Python APIs for easy integration of pipelines into personal projects. Before use, you need to install PaddleX. For installation instructions, please refer to [PaddleX Local Installation Tutorial](../installation/installation.en.md). This tutorial introduces the usage of the pipeline tool with an example of detecting anomalies in equipment nodes.

## 1. Select a Pipeline
First, choose the corresponding PaddleX pipeline based on your task scenario. This task aims to identify and mark abnormal behaviors or states in equipment nodes, helping enterprises and organizations promptly detect and resolve issues in application server nodes, thereby improving system reliability and availability. Recognizing this as a time series anomaly detection task, we will use PaddleX's Time Series Anomaly Detection Pipeline. If you are unsure about the correspondence between tasks and pipelines, you can refer to the [PaddleX Pipeline List (CPU/GPU)](../support_list/models_list.en.md) for an overview of pipeline capabilities.

## 2. Quick Experience
PaddleX offers two ways to experience its capabilities: locally on your machine or on the <b>Baidu AIStudio Community</b>.

* Local Experience:

```python
from paddlex import create_model
model = create_model("PatchTST_ad")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
* AIStudio Community Experience: Visit the [Official Time Series Anomaly Detection Application](https://aistudio.baidu.com/community/app/105708/webUI?source=appCenter) to experience the capabilities of time series anomaly detection tasks.
Note: Due to the tight correlation between time series data and scenarios, the online experience of official models for time series tasks is tailored to a specific scenario and is not a general solution. Therefore, the experience mode does not support using arbitrary files to evaluate the official model's performance. However, after training a model with your own scenario data, you can select your trained model and use data from the corresponding scenario for online experience.

## 3. Choose a Model
PaddleX provides 4 end-to-end time series anomaly detection models. For details, refer to the [Model List](../support_list/models_list.en.md). The benchmarks of these models are as follows:

<table>
<thead>
<tr>
<th>Model Name</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-Score</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>DLinear_ad</td>
<td>0.9898</td>
<td>0.9396</td>
<td>0.9641</td>
<td>72.8K</td>
<td>A simple, efficient, and easy-to-use time series anomaly detection model</td>
</tr>
<tr>
<td>Nonstationary_ad</td>
<td>0.9855</td>
<td>0.8895</td>
<td>0.9351</td>
<td>1.5MB</td>
<td>A transformer-based model optimized for anomaly detection in non-stationary time series</td>
</tr>
<tr>
<td>AutoEncoder_ad</td>
<td>0.9936</td>
<td>0.8436</td>
<td>0.9125</td>
<td>32K</td>
<td>A classic autoencoder-based model that is efficient and easy to use for time series anomaly detection</td>
</tr>
<tr>
<td>PatchTST_ad</td>
<td>0.9878</td>
<td>0.9070</td>
<td>0.9457</td>
<td>164K</td>
<td>A high-precision time series anomaly detection model that balances local patterns and global dependencies</td>
</tr>
</tbody>
</table>
> <b>Note: The above accuracy metrics are measured on the [PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar) dataset with a time series length of 100.</b>

## 4. Data Preparation and Validation
### 4.1 Data Preparation

To demonstrate the entire process of time series anomaly detection, we will use the publicly available MSL (Mars Science Laboratory) dataset for model training and validation. The PSM (Planetary Science Mission) dataset, sourced from NASA, comprises 55 dimensions and includes telemetry anomaly data reported by the spacecraft's monitoring system for unexpected event anomalies (ISA). With its practical application background, it better reflects real-world anomaly scenarios and is commonly used to test and validate the performance of time series anomaly detection models. This tutorial will perform anomaly detection based on this dataset.

We have converted the dataset into a standard data format, and you can obtain a sample dataset using the following command. For an introduction to the data format, please refer to the [Time Series Anomaly Detection Module Development Tutorial](../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.en.md).


You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/msl.tar -P ./dataset
tar -xf ./dataset/msl.tar -C ./dataset/
```

<b>Data Considerations</b>
 * Time series anomaly detection is an unsupervised learning task, thus labeled training data is not required. The collected training samples should ideally consist solely of normal data, i.e., devoid of anomalies, with the label column in the training set set to 0 or, alternatively, the label column can be omitted entirely. For the validation set, to assess accuracy, labeling is necessary. Points that are anomalous at a particular timestamp should have their labels set to 1, while normal points should have labels of 0.
 * Handling Missing Values: To ensure data quality and integrity, missing values can be imputed based on expert knowledge or statistical methods.
 * Non-Repetitiveness: Ensure that data is collected in chronological order by row, with no duplication of timestamps.

### 4.2 Data Validation
Data Validation can be completed with just one command:

```
python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/msl
```
After executing the above command, PaddleX will validate the dataset, summarize its basic information, and print `Check dataset passed !` in the log if the command runs successfully. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the current directory's `./output/check_dataset` directory, including example time series data and class distribution histograms.

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 58317,
    "train_table": [
      [
        "timestamp",
        "0",
        "1",
        "2",
        "..."
      ],
      [
        "..."
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": "./dataset/msl",
  "show_type": "csv",
  "dataset_type": "TSADDataset"
}
```
The above verification results have omitted some data parts. `check_pass` being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.train_samples`: The number of samples in the training set of this dataset is 58317.
* `attributes.val_samples`: The number of samples in the validation set of this dataset is 73729.
* `attributes.train_table`: Sample data rows from the training set of this dataset.
* `attributes.val_table`: Sample data rows from the validation set of this dataset.

<b>Note</b>: Only data that passes the verification can be used for training and evaluation.

### 4.3 Dataset Format Conversion/Dataset Splitting (Optional)
If you need to convert the dataset format or re-split the dataset, refer to Section 4.1.3 in the [Time Series Anomaly Detection Module Development Tutorial](../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.en.md).

## 5. Model Training and Evaluation
### 5.1 Model Training
Before training, ensure that you have verified the dataset. To complete PaddleX model training, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
-o Global.mode=train \
-o Global.dataset_dir=./dataset/msl \
-o Train.epochs_iters=5 \
-o Train.batch_size=16 \
-o Train.learning_rate=0.0001 \
-o Train.time_col=timestamp \
-o Train.feature_cols=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54 \
-o Train.freq=1 \
-o Train.label_col=label \
-o Train.seq_len=96
```
PaddleX supports modifying training hyperparameters and single-machine single-GPU training (time series models only support single-GPU training). Simply modify the configuration file or append command-line parameters.

Each model in PaddleX provides a configuration file for model development to set relevant parameters. Model training-related parameters can be set by modifying the `Train` fields in the configuration file. Some example explanations for parameters in the configuration file are as follows:

* `Global`:
  * `mode`: Mode, supporting dataset verification (`check_dataset`), model training (`train`), model evaluation (`evaluate`), and single-case testing (`predict`).
  * `device`: Training device, options include `cpu` and `gpu`. You can check the models supported on different devices in the [PaddleX Model List (CPU/GPU)](../support_list/models_list.en.md) document at the same level directory.
* `Train`: Training hyperparameter settings;
  * `epochs_iters`: Number of training epochs.
  * `learning_rate`: Training learning rate.
  * `batch_size`: Training batch size for a single GPU.
  * `time_col`: Time column, set the column name of the time series dataset's time column based on your data.
  * `feature_cols`: Feature variables indicating variables related to whether the device is abnormal.
  * `freq`: Frequency of the time series dataset.
  * `input_len`: The length of the time series input to the model. The time series will be sliced according to this length, and the model will predict whether there is an anomaly in this segment of the time series for that length. The recommended input length should be considered in the context of the actual scenario. In this tutorial, the input length is 96, which means we hope to predict whether there are anomalies at 96 time points.
  * `label`: Represents the number indicating whether a time point in the time series is abnormal. Anomalous points are labeled as 1, and normal points are labeled as 0. In this tutorial, the anomaly monitoring dataset uses label for this purpose.

For more introductions to hyperparameters, please refer to [PaddleX Time Series Task Model Configuration File Parameter Instructions](../module_usage/instructions/config_parameters_time_series.en.md). The above parameters can be set by appending command-line arguments. For example, to specify the mode as model training: `-o Global.mode=train`; to specify the first GPU for training: `-o Global.device=gpu:0`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`.

<details><summary> More Details (Click to Expand)  </summary>

<ul>
<li>During the model training process, PaddleX automatically saves the model weight files, with the default directory being output. If you need to specify a different save path, you can configure it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts away the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights are produced simultaneously. By default, static graph weights are selected for inference.</li>
<li>When training other models, you need to specify the corresponding configuration file. The correspondence between models and configuration files can be found in the <a href="../support_list/models_list.en.md">PaddleX Model List (CPU/GPU)</a></li>
</ul>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including the following:</p>
<p><b>Explanation of Training Outputs:</b></p>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including the following:</p>
<p><code>train_result.json</code>: A training result record file that logs whether the training task was completed normally, as well as the output weight metrics, relevant file paths, etc.
<code>train.log</code>: A training log file that records changes in model metrics, loss values, and other information during the training process.
<code>config.yaml</code>: The training configuration file that records the hyperparameter configurations for this training session.
<code>best_accuracy.pdparams.tar</code>, <code>scaler.pkl</code>, <code>.checkpoints</code>, <code>.inference*</code>: Files related to model weights, including network parameters, optimizers, static graph network parameters, etc.</p></details>

### 5.2 Model Evaluation
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation requires just one command:

```bash
python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/msl
```
Similar to model training, model evaluation supports setting through modifying the configuration file or appending command-line parameters.

<b>Note</b>: When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command-line parameter, such as `-o Evaluate.weight_path=./output/best_model/model.pdparams`.

After completing the model evaluation, typically, the following outputs are generated:

Upon completion of model evaluation, `evaluate_result.json` will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully, and the model's evaluation metrics, including F1, recall, and precision.

### 5.3 Model Tuning
After learning about model training and evaluation, we can improve the model's accuracy by adjusting hyperparameters. By reasonably adjusting the number of training epochs, you can control the depth of model training, avoiding overfitting or underfitting. The setting of the learning rate is related to the speed and stability of model convergence. Therefore, when optimizing model performance, it is essential to carefully consider the values of these two parameters and adjust them flexibly based on actual conditions to achieve the best training effect.

Based on the method of controlled variables, we can adopt the following approach for hyperparameter tuning in time-series anomaly detection:

It is recommended to follow the method of controlled variables when debugging parameters:

1. First, fix the number of training epochs to 5, batch size to 16, and input length to 96.
2. Initiate three experiments based on the PatchTST_ad model with learning rates of: 0.0001, 0.0005, 0.001.
3. It can be found that Experiment 3 has the highest accuracy, with a learning rate of 0.001. Therefore, fix the learning rate at 0.001 and try increasing the number of training epochs to 20.
4. It can be found that the accuracy of Experiment 4 is the same as that of Experiment 3, indicating that there is no need to further increase the number of training epochs.

Learning Rate Exploration Results:

<table>
<thead>
<tr>
<th>Experiment</th>
<th>Epochs</th>
<th>Learning Rate</th>
<th>Batch Size</th>
<th>Input Length</th>
<th>Training Environment</th>
<th>Validation F1 Score (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Experiment 1</td>
<td>5</td>
<td>0.0001</td>
<td>16</td>
<td>96</td>
<td>1 GPU</td>
<td>79.5</td>
</tr>
<tr>
<td>Experiment 2</td>
<td>5</td>
<td>0.0005</td>
<td>16</td>
<td>96</td>
<td>1 GPU</td>
<td>80.1</td>
</tr>
<tr>
<td>Experiment 3</td>
<td>5</td>
<td>0.001</td>
<td>16</td>
<td>96</td>
<td>1 GPU</td>
<td>80.9</td>
</tr>
</tbody>
</table>
Increasing Training Epochs Results:

<table>
<thead>
<tr>
<th>Experiment</th>
<th>Epochs</th>
<th>Learning Rate</th>
<th>Batch Size</th>
<th>Input Length</th>
<th>Training Environment</th>
<th>Validation F1 Score (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Experiment 3</td>
<td>5</td>
<td>0.0005</td>
<td>16</td>
<td>96</td>
<td>1 GPU</td>
<td>80.9</td>
</tr>
<tr>
<td>Experiment 4</td>
<td>20</td>
<td>0.0005</td>
<td>16</td>
<td>96</td>
<td>1 GPU</td>
<td>80.9</td>
</tr>
</tbody>
</table>
## 6. Production Line Testing
Replace the model in the production line with the fine-tuned model for testing, using the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv) for prediction:

```
python main.py -c paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv"
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `PatchTST_ad.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../module_usage/instructions/config_parameters_time_series.en.md).

## 7. Integration/Deployment
If the general-purpose time series anomaly detection pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

1. If you need to use the fine-tuned model weights, you can obtain the ts_anomaly_detection production configuration file and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config ts_anomaly_detection --save_path ./my_path
```

Fill in the local path of the fine-tuned model weights into the `model_dir` in the production configuration file. If you need to directly apply the general time-series classification pipeline in your Python project, you can refer to the following example:

```yaml
pipeline_name: ts_anomaly_detection

SubModules:
  TSAnomalyDetection:
    module_name: ts_anomaly_detection
    model_name: PatchTST_ad
    model_dir: ./output/inference  # 此处替换为您训练后得到的模型权重本地路径
    batch_size: 1
```

Then, execute the following code to complete the prediction:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/ts_anomaly_detection.yaml")
output = pipeline.predict("pre_ts.csv")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_csv("./output/") # 保存csv格式结果
```

For more parameters, please refer to the [Time Series Anomaly Detection Pipeline Usage Tutorial](../pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.en.md)

1. Additionally, PaddleX's time series anomaly detection pipeline also offers a service-oriented deployment method, detailed as follows:

Service-Oriented Deployment: This is a common deployment form in actual production environments. By encapsulating the inference functionality as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving service-oriented deployment of pipelines at low cost. For detailed instructions on service-oriented deployment, please refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/serving.en.md).
You can choose the appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.
