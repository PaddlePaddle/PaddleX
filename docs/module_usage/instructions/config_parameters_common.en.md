---
comments: true
---

# PaddleX Common Model Configuration File Parameter Explanation

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
<td>Whether to convert the dataset format; Image classification, pedestrian attribute recognition, vehicle attribute recognition, document orientation classification, object detection, pedestrian detection, vehicle detection, face detection, anomaly detection, text detection, seal text detection, text recognition, table recognition, image rectification, and layout area detection do not support data format conversion; Image multi-label classification supports COCO format conversion; Image feature, semantic segmentation, and instance segmentation support LabelMe format conversion; Object detection and small object detection support VOC and LabelMe format conversion; Formula recognition supports PKL format conversion; Time series prediction, time series anomaly detection, and time series classification support xlsx and xls format conversion</td>
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
<tr>
<td>split.gallery_percent</td>
<td>int</td>
<td>Sets the percentage of gallery samples in the validation set, an integer between 0-100, ensuring the sum with train_percent and query_percent is 100; This parameter is only used in the image feature module</td>
<td>null</td>
</tr>
<tr>
<td>split.query_percent</td>
<td>int</td>
<td>Sets the percentage of query samples in the validation set, an integer between 0-100, ensuring the sum with train_percent and gallery_percent is 100; This parameter is only used in the image feature module</td>
<td>null</td>
</tr>
</tbody>
</table>

# Train

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
<td>num_classes</td>
<td>int</td>
<td>Number of classes in the dataset; If you need to train on a private dataset, you need to set this parameter; Image rectification, text detection, seal text detection, text recognition, formula recognition, table recognition, time series prediction, time series anomaly detection, and time series classification do not support this parameter</td>
<td>Number of classes specified in the YAML file</td>
</tr>
<tr>
<td>epochs_iters</td>
<td>int</td>
<td>Number of times the model repeats learning the training data</td>
<td>Number of iterations specified in the YAML file</td>
</tr>
<tr>
<td>batch_size</td>
<td>int</td>
<td>Training batch size</td>
<td>Training batch size specified in the YAML file</td>
</tr>
<tr>
<td>learning_rate</td>
<td>float</td>
<td>Initial learning rate</td>
<td>Initial learning rate specified in the YAML file</td>
</tr>
<tr>
<td>pretrain_weight_path</td>
<td>str</td>
<td>Pre-trained weight path</td>
<td>null</td>
</tr>
<tr>
<td>warmup_steps</td>
<td>int</td>
<td>Warm-up steps</td>
<td>Warm-up steps specified in the YAML file</td>
</tr>
<tr>
<td>resume_path</td>
<td>str</td>
<td>Model resume path after interruption</td>
<td>null</td>
</tr>
<tr>
<td>log_interval</td>
<td>int</td>
<td>Training log printing interval</td>
<td>Training log printing interval specified in the YAML file</td>
</tr>
<tr>
<td>eval_interval</td>
<td>int</td>
<td>Model evaluation interval</td>
<td>Model evaluation interval specified in the YAML file</td>
</tr>
<tr>
<td>save_interval</td>
<td>int</td>
<td>Model saving interval; not supported for anomaly detection, semantic segmentation, image rectification, time series forecasting, time series anomaly detection, and time series classification</td>
<td>Model saving interval specified in the YAML file</td>
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
<tr>
<td>log_interval</td>
<td>int</td>
<td>Evaluation log printing interval</td>
<td>Evaluation log printing interval specified in the YAML file</td>
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
<td>kernel_option</td>
<td>dict</td>
<td>Path to the prediction input</td>
<td>Inference engine setting, such as: "run_mode: paddle"</td>
</tr>

</tbody>
</table>
