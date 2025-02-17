---
comments: true
---

# PaddleX 3D Task Model Configuration File Parameters Explanation

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
<tr>
<td>load_cam_from</td>
<td>str</td>
<td>Pre-trained parameter path of cam branch</td>
<td>The pre-trained parameter path of the cam branch specified in the YAML file</td>
</tr>
<tr>
<td>load_lidar_from</td>
<td>str</td>
<td>Pre-trained parameter path of lidar branch</td>
<td>The pre-trained parameter path of the lidar branch specified in the YAML file</td>
</tr>
<tr>
<td>datart_prefix</td>
<td>bool</td>
<td>Whether the data set path is prefixed</td>
<td>True</td>
</tr>
<tr>
<td>version</td>
<td>str</td>
<td>Dataset version</td>
<td>"mini"</td>
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
<td>Whether to convert the dataset format;</td>
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
### Common Parameters for 3D Tasks
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
<td>warmup_steps</td>
<td>str</td>
<td>The number of steps performed in the warm-up training</td>
<td>The number of warm-up steps specified in the YAML file</td>
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
<td>batch_size</td>
<td>int</td>
<td>Batch size</td>
<td>Batch size specified in the YAML file
</td>
</tr>
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
<td>kernel_option.run_mode</td>
<td>str</td>
<td>The low-level implementation method of the operator</td>
<td>The underlying implementation method specified in the yaml file</td>
</tr>
</tbody>
</table>
