---
comments: true
---

# PaddleX Time Series Task Model Configuration File Parameters Explanation

# Global

<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>model</td>
<td>str</td>
<td>Specifies the model name</td>
<td>Model name specified in the YAML file</td>
</tr>
<tr>
<td>mode</td>
<td>str</td>
<td>Specifies the mode (check_dataset/train/evaluate/export/predict)</td>
<td>check_dataset</td>
</tr>
<tr>
<td>dataset_dir</td>
<td>str</td>
<td>Path to the dataset</td>
<td>Dataset path specified in the YAML file</td>
</tr>
<tr>
<td>device</td>
<td>str</td>
<td>Specifies the device to use</td>
<td>Device ID specified in the YAML file</td>
</tr>
<tr>
<td>output</td>
<td>str</td>
<td>Output path</td>
<td>"output"</td>
</tr>
</tbody>
</table>
# CheckDataset

<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>convert.enable</td>
<td>bool</td>
<td>Whether to convert the dataset format; time series prediction, anomaly detection, and classification support data conversion from xlsx and xls formats</td>
<td>False</td>
</tr>
<tr>
<td>convert.src_dataset_type</td>
<td>str</td>
<td>The source dataset format to be converted</td>
<td>null</td>
</tr>
<tr>
<td>split.enable</td>
<td>bool</td>
<td>Whether to re-split the dataset</td>
<td>False</td>
</tr>
<tr>
<td>split.train_percent</td>
<td>int</td>
<td>Sets the percentage of the training set, an integer between 0-100, ensuring the sum with val_percent is 100;</td>
<td>null</td>
</tr>
<tr>
<td>split.val_percent</td>
<td>int</td>
<td>Sets the percentage of the validation set, an integer between 0-100, ensuring the sum with train_percent is 100;</td>
<td>null</td>
</tr>
</tbody>
</table>
# Train
### Common Parameters for Time Series Tasks
<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>epochs_iters</td>
<td>int</td>
<td>The number of times the model repeats learning the training data</td>
<td>Number of iterations specified in the YAML file</td>
</tr>
<tr>
<td>batch_size</td>
<td>int</td>
<td>Batch size</td>
<td>Batch size specified in the YAML file</td>
</tr>
<tr>
<td>learning_rate</td>
<td>float</td>
<td>Initial learning rate</td>
<td>Initial learning rate specified in the YAML file</td>
</tr>
<tr>
<td>time_col</td>
<td>str</td>
<td>Time column, set the column name of the time series dataset's time column based on your data.</td>
<td>Time column specified in the YAML file</td>
</tr>
<tr>
<td>freq</td>
<td>str or int</td>
<td>Frequency, set the time frequency based on your data, e.g., 1min, 5min, 1h.</td>
<td>Frequency specified in the YAML file</td>
</tr>
</tbody>
</table>
### Time Series Forecasting Parameters
<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>target_cols</td>
<td>str</td>
<td>Target variable column(s), set the column name(s) of the target variable(s) in the time series dataset, can be multiple, separated by commas</td>
<td>OT</td>
</tr>
<tr>
<td>input_len</td>
<td>int</td>
<td>For time series forecasting tasks, this parameter represents the length of historical time series input to the model; the input length should be considered in conjunction with the prediction length, generally, the larger the setting, the more historical information can be referenced, and the higher the model accuracy.</td>
<td>96</td>
</tr>
<tr>
<td>predict_len</td>
<td>int</td>
<td>The length of the future sequence that you want the model to predict; the prediction length should be considered in conjunction with the actual scenario, generally, the larger the setting, the longer the future sequence you want to predict, and the lower the model accuracy.</td>
<td>96</td>
</tr>
<tr>
<td>patience</td>
<td>int</td>
<td>Early stopping mechanism parameter, indicating how many times the model's performance on the validation set can be continuously unimproved before stopping training; a larger patience value generally results in longer training time.</td>
<td>10</td>
</tr>
</tbody>
</table>
### Time Series Anomaly Detection
<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>input_len</td>
<td>int</td>
<td>For time series anomaly detection tasks, this parameter represents the length of the time series input to the model, which will slice the time series according to this length to predict whether there is an anomaly in this segment of the time series; the input length should be considered in conjunction with the actual scenario. For example, an input length of 96 indicates that you want to predict whether there are anomalies in 96 time points.</td>
<td>96</td>
</tr>
<tr>
<td>feature_cols</td>
<td>str</td>
<td>Feature variables indicating variables related to whether the device is abnormal, e.g., whether the device is abnormal may be related to the heat dissipation during its operation. Set the column name(s) of the feature variable(s) based on your data, can be multiple, separated by commas.</td>
<td>feature_0,feature_1</td>
</tr>
<tr>
<td>label_col</td>
<td>str</td>
<td>Represents the number indicating whether a time series point is abnormal, with 1 for abnormal points and 0 for normal points.</td>
<td>label</td>
</tr>
</tbody>
</table>
### Time Series Classification
<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>target_cols</td>
<td>str</td>
<td>Feature variable columns used for category discrimination. You need to set the column names of the target variables in the time series dataset based on your own data. It can be multiple, separated by commas.</td>
<td>dim_0,dim_1,dim_2</td>
</tr>
<tr>
<td>freq</td>
<td>str or int</td>
<td>Frequency, which needs to be set based on your own data. Examples of time frequencies include: 1min, 5min, 1h.</td>
<td>1</td>
</tr>
<tr>
<td>group_id</td>
<td>str</td>
<td>A group ID represents a time series sample. Time series sequences with the same ID constitute a sample. Set the column name of the specified group ID based on your own data, e.g., group_id.</td>
<td>group_id</td>
</tr>
<tr>
<td>static_cov_cols</td>
<td>str</td>
<td>Represents the category number column of the time series. The labels of the same sample are the same. Set the column name of the category based on your own data, e.g., label.</td>
<td>label</td>
</tr>
</tbody>
</table>
# Evaluate
<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>weight_path</td>
<td>str</td>
<td>Evaluation model path</td>
<td>Default local path from training output, when specified as None, indicates using official weights</td>
</tr>
</tbody>
</table>
# Export
<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>weight_path</td>
<td>str</td>
<td>Dynamic graph weight path for exporting the model</td>
<td>Default local path from training output, when specified as None, indicates using official weights</td>
</tr>
</tbody>
</table>
# Predict
<table>
<thead>
<tr>
<th>Parameter Name</th>
<th>Data Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>batch_size</td>
<td>int</td>
<td>Prediction batch size</td>
<td>The prediction batch size specified in the YAML file</td>
</tr>
<tr>
<td>model_dir</td>
<td>str</td>
<td>Path to the prediction model</td>
<td>The default local inference model path produced by training. When specified as None, it indicates the use of official weights</td>
</tr>
<tr>
<td>input</td>
<td>str</td>
<td>Path to the prediction input</td>
<td>The prediction input path specified in the YAML file</td>
</tr>
<tr>
<td>use_hpip</td>
<td>bool</td>
<td>Whether to enable the high-performance inference plugin</td>
<td></td>
</tr>
<tr>
<td>hpip_config</td>
<td>dict | None</td>
<td>High-performance inference configuration</td>
<td></td>
</tr>
</tbody>
</table>
