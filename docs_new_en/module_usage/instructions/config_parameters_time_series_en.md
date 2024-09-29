# PaddleX Time Series Task Model Configuration File Parameters Explanation

# Global

| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| --- | --- | --- | --- | --- |  
| model | str | Specifies the model name | - | Required |  
| mode | str | Specifies the mode (check_dataset/train/evaluate/export/predict) | - | Required |  
| dataset_dir | str | Path to the dataset | - | Required |  
| device | str | Specifies the device to use | - | Required |  
| output | str | Output directory path | "output" | Optional |

# CheckDataset

| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| --- | --- | --- | --- | --- |  
| convert.enable | bool | Whether to enable dataset format conversion | False | Optional |  
| convert.src_dataset_type | str | The source dataset format to convert from | null | Required |  
| split.enable | bool | Whether to re-split the dataset | False | Optional |  
| split.train_percent | int | Sets the percentage of the training set, an integer between 0-100. It should sum up to 100 with `val_percent`. | - | Optional |  
| split.val_percent | int | Sets the percentage of the validation set, an integer between 0-100. It should sum up to 100 with `train_percent`. | - | Optional |  
  
# Train

### Common parameters for time series tasks

| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| --- | --- | --- | --- | --- |  
| epochs_iters | int | Number of times the model learns from the training data | - | Required |  
| batch_size | int | Batch size for training | - | Required |  
| learning_rate | float | Initial learning rate | - | Required |  
| time_col | str | Time column, must be set to the column name that represents the time series data's timestamp in your dataset. | - | Required |  
| freq | str or int | Frequency, must be set to the time frequency of your data, such as '1min', '5min', '1h'. | - | Required |  
  
**Note**: The default values for these parameters are not specified ("-"), indicating that they must be explicitly provided by the user based on their specific dataset and requirements.

### Time series forecasting parameters


| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| --- | --- | --- | --- | --- |  
| target_cols | str | Target variable column(s), must be set to the column name(s) that represent the target variable(s) in your time series dataset. Multiple columns can be specified by separating them with commas. | - | Required |  
| input_len | int | For time series prediction tasks, this parameter represents the length of historical time series data input to the model. The input length should be considered in conjunction with the prediction length and the specific scenario. Generally, a larger input length allows the model to reference more historical information, which may lead to higher accuracy. | - | Required |  
| predict_len | int | The desired length of the future sequence that the model should predict. The prediction length should be considered in conjunction with the specific scenario. Generally, a larger prediction length means predicting a longer future sequence, which may lead to lower model accuracy. | - | Required |  
| patience | int | A parameter for the early stopping mechanism, indicating how many times the model's performance on the validation set can be consecutively unchanged before stopping training. A larger patience value generally results in longer training time. | - | Required |  
  
**Note**: The default values for these parameters are not specified ("-"), indicating that they must be explicitly provided by the user based on their specific dataset and requirements.

### Time series anomaly detection parameters

| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| --- | --- | --- | --- | --- |  
| input_len | int | For time series anomaly detection tasks, this parameter represents the length of the time series input to the model. The time series will be sliced according to this length, and the model will predict whether there are anomalies within this segment. The input length should be considered based on the specific scenario. For example, an input length of 96 indicates the desire to predict whether there are anomalies at 96 time points. | - | Required |  
| feature_cols | str | Feature columns represent variables that can be used to determine whether a device is anomalous. For instance, whether a device is anomalous may be related to the amount of heat it generates during operation. Based on your data, set the column names of the feature variables. Multiple columns can be specified by separating them with commas. | - | Required |  
| label_col | str | Represents the label indicating whether a time series point is anomalous. Anomalous points are labeled as 1, and normal points are labeled as 0. | - | Required |  
  
**Note**: The default values for these parameters are not specified ("-"), indicating that they must be explicitly provided by the user based on their specific dataset and requirements. 

### Time series classification parameters

| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| --- | --- | --- | --- | --- |  
| num_classes | int | The number of classes in the dataset. | - | Required |  
| target_cols | str | The column(s) of the feature variable used to determine the class, which must be set according to your dataset in the time series dataset. Multiple columns can be specified by separating them with commas. | - | Required |  
| freq | str or int | The frequency of the time series, which must be set according to your data. Examples include '1min', '5min', '1h'. | - | Required |  
| group_id | str | A group ID represents a time series sample. Time series sequences with the same ID constitute a sample. Set the column name for the specified group ID according to your data, e.g., 'group_id'. | - | Required |  
| static_cov_cols | str | Represents the class ID column for the time series. Samples within the same class share the same label. Set the column name for the class according to your data, e.g., 'label'. | - | Required |  
  
**Note**: The default values for these parameters are not specified ("-"), indicating that they must be explicitly provided by the user based on their specific dataset and requirements. 

# Evaluate

| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| --- | --- | --- | --- | --- |  
| weight_path | str | The path to the model weights for evaluation. | - | Required |  

# Export

| Parameter Name | Data Type | Description | Default Value | Required/Optional |    
| -------------- | --------- | -------------------------------------------------- | ------------------- | ------------- |    
| weight_path    | str       | The path to the dynamic graph weight file used for exporting the model |The official dynamic graph weight URLs for each model. | Required      |    
  
# Predict

| Parameter Name | Data Type | Description | Default Value | Required/Optional |  
| -------------- | --------- | ---------------------------------- | --------------- | ------------- |    
| model_dir      | str       | Path to the directory containing the prediction model |The official weight | Optional      |  
| input          | str       | Path to the input data for prediction | (No default, user must specify) | Required      |  
| batch_size     | int       | The number of samples processed in each prediction batch | (No default, user must specify) | Required      |  

